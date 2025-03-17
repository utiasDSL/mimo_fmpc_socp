'''NL Model Predictive Safety Certification (NL MPSC).

The core idea is that any learning controller input can be either certificated as safe or, if not safe, corrected
using an MPC controller based on Robust NL MPC.

Based on
    * K.P. Wabsersich and M.N. Zeilinger 'Linear model predictive safety certification for learning-based control' 2019
      https://arxiv.org/pdf/1803.08552.pdf
    * J. Köhler, R. Soloperto, M. A. Müller, and F. Allgöwer, “A computationally efficient robust model predictive
      control framework for uncertain nonlinear systems -- extended version,” IEEE Trans. Automat. Contr., vol. 66,
      no. 2, pp. 794 801, Feb. 2021, doi: 10.1109/TAC.2020.2982585. http://arxiv.org/abs/1910.12081
'''

import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template.acados_model import AcadosModel
from scipy.linalg import block_diag

from safe_control_gym.controllers.mpc.mpc_utils import discretize_linear_system, rk_discrete
from safe_control_gym.safety_filters.mpsc.mpsc import MPSC
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function


class NL_MPSC(MPSC):
    '''Model Predictive Safety Certification Class.'''

    def __init__(self,
                 env_func,
                 horizon: int = 10,
                 q_mpc: list = None,
                 r_mpc: list = None,
                 integration_algo: str = 'rk4',
                 warmstart: bool = True,
                 additional_constraints: list = None,
                 use_terminal_set: bool = True,
                 n_samples: int = 600,
                 cost_function: Cost_Function = Cost_Function.ONE_STEP_COST,
                 mpsc_cost_horizon: int = 5,
                 decay_factor: float = 0.85,
                 soften_constraints: bool = False,
                 slack_cost: float = 250,
                 max_w: float = 0.002,
                 **kwargs
                 ):
        '''Initialize the MPSC.

        Args:
            env_func (partial BenchmarkEnv): Environment for the task.
            horizon (int): The MPC horizon.
            integration_algo (str): The algorithm used for integrating the dynamics,
                either 'rk4', 'rk', or 'cvodes'.
            warmstart (bool): If the previous MPC soln should be used to warmstart the next mpc step.
            additional_constraints (list): List of additional constraints to consider.
            use_terminal_set (bool): Whether to use a terminal set constraint or not.
            n_samples (int): The number of state/action pairs to test when determining w_func.
            cost_function (Cost_Function): A string (from Cost_Function) representing the cost function to be used.
            mpsc_cost_horizon (int): How many steps forward to check for constraint violations.
            decay_factor (float): How much to discount future costs.
        '''

        self.model_bias = None
        super().__init__(env_func, horizon, q_mpc, r_mpc, integration_algo, warmstart, additional_constraints, use_terminal_set, cost_function, mpsc_cost_horizon, decay_factor, **kwargs)

        self.n_samples = n_samples
        self.soften_constraints = soften_constraints
        self.slack_cost = slack_cost
        self.max_w = max_w

        self.n = self.model.nx
        self.m = self.model.nu
        self.q = self.model.nx

        self.state_constraint = self.constraints.state_constraints[0]
        self.input_constraint = self.constraints.input_constraints[0]

        [self.X_mid, L_x, l_x] = self.box2polytopic(self.state_constraint)
        [self.U_mid, L_u, l_u] = self.box2polytopic(self.input_constraint)

        # number of constraints
        p_x = l_x.shape[0]
        p_u = l_u.shape[0]
        self.p = p_x + p_u

        self.L_x = np.vstack((L_x, np.zeros((p_u, self.n))))
        self.L_u = np.vstack((np.zeros((p_x, self.m)), L_u))
        self.l_xu = np.concatenate([l_x, l_u])

        self.setup_optimizer()

    def set_dynamics(self):
        '''Compute the discrete dynamics.'''

        if self.integration_algo == 'LTI':
            dfdxdfdu = self.model.df_func(x=self.X_EQ, u=self.U_EQ)
            self.Ac = dfdxdfdu['dfdx'].toarray()
            self.Bc = dfdxdfdu['dfdu'].toarray()

            delta_x = self.model.x_sym
            delta_u = self.model.u_sym
            delta_w = cs.MX.sym('delta_w', self.model.nx, 1)

            self.Ad, self.Bd = discretize_linear_system(self.Ac, self.Bc, self.dt, exact=True)

            x_dot_lin_vec = self.Ad @ delta_x + self.Bd @ delta_u

            if self.model_bias is not None:
                x_dot_lin_vec = x_dot_lin_vec + self.model_bias

            dynamics_func = cs.Function('fd',
                                        [delta_x, delta_u],
                                        [x_dot_lin_vec],
                                        ['x0', 'p'],
                                        ['xf'])

            self.Ac = cs.Function('Ac', [delta_x, delta_u, delta_w], [self.Ac], ['x', 'u', 'w'], ['Ac'])
            self.Bc = cs.Function('Bc', [delta_x, delta_u, delta_w], [self.Bc], ['x', 'u', 'w'], ['Bc'])

            self.Ad = cs.Function('Ad', [delta_x, delta_u, delta_w], [self.Ad], ['x', 'u', 'w'], ['Ad'])
            self.Bd = cs.Function('Bd', [delta_x, delta_u, delta_w], [self.Bd], ['x', 'u', 'w'], ['Bd'])
        elif self.integration_algo == 'rk4':
            dynamics_func = rk_discrete(self.model.fc_func,
                                        self.model.nx,
                                        self.model.nu,
                                        self.dt)
        else:
            dynamics_func = cs.integrator('fd', self.integration_algo,
                                          {'x': self.model.x_sym,
                                           'p': self.model.u_sym,
                                           'ode': self.model.x_dot}, {'tf': self.dt}
                                          )

        self.dynamics_func = dynamics_func

    def box2polytopic(self, constraint):
        '''Convert constraints into an explicit polytopic form. This assumes that constraints contain the origin.

        Args:
            constraint (Constraint): The constraint to be converted.

        Returns:
            L (ndarray): The polytopic matrix.
            l (ndarray): Whether the constraint is active.
        '''

        Limit = []
        limit_active = []

        Z_mid = (constraint.upper_bounds + constraint.lower_bounds) / 2.0
        Z_limits = np.array([[constraint.upper_bounds[i] - Z_mid[i], constraint.lower_bounds[i] - Z_mid[i]] for i in range(constraint.upper_bounds.shape[0])])

        dim = Z_limits.shape[0]
        eye_dim = np.eye(dim)

        for constraint_id in range(0, dim):
            if Z_limits[constraint_id, 0] != -float('inf'):
                if Z_limits[constraint_id, 0] == 0:
                    limit_active += [0]
                    Limit += [-eye_dim[constraint_id, :]]
                else:
                    limit_active += [1]
                    factor = 1 / Z_limits[constraint_id, 0]
                    Limit += [factor * eye_dim[constraint_id, :]]

            if Z_limits[constraint_id, 1] != float('inf'):
                if Z_limits[constraint_id, 1] == 0:
                    limit_active += [0]
                    Limit += [eye_dim[constraint_id, :]]
                else:
                    limit_active += [1]
                    factor = 1 / Z_limits[constraint_id, 1]
                    Limit += [factor * eye_dim[constraint_id, :]]

        return Z_mid, np.array(Limit), np.array(limit_active)

    def setup_casadi_optimizer(self):
        raise NotImplementedError('Casadi not implemented')

    def setup_acados_optimizer(self):
        '''setup_optimizer_acados'''
        # create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # Setup model
        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym
        acados_model.name = self.env.NAME

        acados_model.f_expl_expr = self.model.fc_func(acados_model.x, acados_model.u)
        acados_model.x_labels = self.env.STATE_LABELS
        acados_model.u_labels = self.env.ACTION_LABELS
        acados_model.t_label = 'time'
        ocp.model = acados_model

        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu

        ocp.dims.N = self.horizon

        # set cost module
        ocp.cost.cost_type = 'LINEAR_LS'

        Q_mat = np.zeros((nx, nx))
        R_mat = np.eye(nu)
        ocp.cost.W = block_diag(Q_mat, R_mat)

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:nx + nu, :] = np.eye(nu)

        # Updated on each iteration
        ocp.cost.yref = np.concatenate((self.model.X_EQ, self.model.U_EQ))

        # set constraints
        ocp.constraints.constr_type = 'BGH'
        ocp.constraints.x0 = self.model.X_EQ
        ocp.constraints.C = self.L_x
        ocp.constraints.D = self.L_u
        ocp.constraints.lg = -1000 * np.ones((self.p))
        ocp.constraints.ug = np.zeros((self.p))

        # Slack
        if self.soften_constraints:
            ocp.constraints.Jsg = np.eye(self.p)
            ocp.cost.Zu = self.slack_cost * np.ones(self.p)
            ocp.cost.Zl = self.slack_cost * np.ones(self.p)
            ocp.cost.zl = self.slack_cost * np.ones(self.p)
            ocp.cost.zu = self.slack_cost * np.ones(self.p)

        # Options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.hpipm_mode = 'BALANCE'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 200

        # set prediction horizon
        ocp.solver_options.tf = self.dt * self.horizon

        solver_json = 'acados_ocp_mpsf.json'
        ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json, generate=True, build=True)

        for stage in range(self.mpsc_cost_horizon):
            ocp_solver.cost_set(stage, 'W', (self.cost_function.decay_factor**stage) * ocp.cost.W)

        for stage in range(self.mpsc_cost_horizon, self.horizon):
            ocp_solver.cost_set(stage, 'W', 0 * ocp.cost.W)

        g = np.zeros((self.horizon, self.p))

        for i in range(self.horizon):
            for j in range(self.p):
                tighten_by = (self.max_w * i) if j < self.n * 2 else 0
                g[i, j] = (self.l_xu[j] - tighten_by)
            g[i, :] += (self.L_x @ self.X_mid) + (self.L_u @ self.U_mid)
            ocp_solver.constraints_set(i, 'ug', g[i, :])

        self.ocp_solver = ocp_solver
