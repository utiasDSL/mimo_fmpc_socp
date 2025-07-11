'''Model Predictive Control.'''
from copy import deepcopy

import casadi as cs
import numpy as np
from termcolor import colored

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.controllers.mpc.mpc_utils import (compute_discrete_lqr_gain_from_cont_linear_system,
                                                        compute_state_rmse, get_cost_weight_matrix,
                                                        reset_constraints, rk_discrete)
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS, create_constraint_list
from safe_control_gym.utils.utils import timing
from numpy.linalg import LinAlgError

class MPC(BaseController):
    '''MPC with full nonlinear model.'''

    def __init__(
            self,
            env_func,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            warmstart: bool = True,
            soft_constraints: bool = False,
            soft_penalty: float = 10000,
            terminate_run_on_done: bool = True,
            constraint_tol: float = 1e-6,
            # runner args
            # shared/base args
            output_dir: str = 'results/temp',
            additional_constraints: list = None,
            use_gpu: bool = False,
            seed: int = 0,
            compute_initial_guess_method: str = 'ipopt',
            use_lqr_gain_and_terminal_cost: bool = False,
            init_solver: str = 'ipopt',
            solver: str = 'ipopt',
            **kwargs
    ):
        '''Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            horizon (int): mpc planning horizon.
            q_mpc (list): diagonals of state cost weight.
            r_mpc (list): diagonals of input/action cost weight.
            warmstart (bool): if to initialize from previous iteration.
            soft_constraints (bool): Formulate the constraints as soft constraints.
            terminate_run_on_done (bool): Terminate the run when the environment returns done or not.
            constraint_tol (float): Tolerance to add the the constraint as sometimes solvers are not exact.
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): List of additional constraints
            use_gpu (bool): False (use cpu) True (use cuda).
            seed (int): random seed.
            compute_initial_guess_method (str): Method to compute the initial guess. Options: None, 'ipopt', 'lqr'.
            use_lqr_gain_and_terminal_cost (bool): Use the LQR ancillary gain and terminal cost in the MPC.
            init_solver (str): Solver to use for initial guess computation.
            solver (str): Solver to use for MPC optimization.
        '''
        super().__init__(env_func=env_func, output_dir=output_dir, use_gpu=use_gpu, seed=seed, **kwargs)
        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__.update({k: v})

        # Task.
        self.env = env_func()
        if additional_constraints is not None:
            additional_ConstraintsList = create_constraint_list(additional_constraints,
                                                                GENERAL_CONSTRAINTS,
                                                                self.env)
            self.additional_constraints = additional_ConstraintsList.constraints
            self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(self.env.constraints.constraints + self.additional_constraints)
        else:
            self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(self.env.constraints.constraints)
            self.additional_constraints = []
        # Model parameters
        self.model = self.get_prior(self.env)
        self.dt = self.model.dt
        self.T = horizon
        self.Q = get_cost_weight_matrix(self.q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_mpc, self.model.nu)

        self.constraint_tol = constraint_tol
        self.soft_constraints = soft_constraints
        self.soft_penalty = soft_penalty
        self.warmstart = warmstart
        self.terminate_run_on_done = terminate_run_on_done

        self.X_EQ = self.env.X_GOAL
        self.U_EQ = self.env.U_GOAL
        self.compute_initial_guess_method = compute_initial_guess_method
        self.use_lqr_gain_and_terminal_cost = use_lqr_gain_and_terminal_cost
        self.init_solver = init_solver
        self.solver = solver

    def add_constraints(self,
                        constraints
                        ):
        '''Add the constraints (from a list) to the system.

        Args:
            constraints (list): List of constraints controller is subject too.
        '''
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(constraints + self.constraints.constraints)

    def remove_constraints(self,
                           constraints
                           ):
        '''Remove constraints from the current constraint list.

        Args:
            constraints (list): list of constraints to be removed.
        '''
        old_constraints_list = self.constraints.constraints
        for constraint in constraints:
            assert constraint in self.constraints.constraints, \
                ValueError('This constraint is not in the current list of constraints')
            old_constraints_list.remove(constraint)
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(old_constraints_list)

    def close(self):
        '''Cleans up resources.'''
        self.env.close()

    def reset(self):
        '''Prepares for training or evaluation.'''
        print(colored('Resetting MPC', 'green'))
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            # Step along the reference.
            self.traj_step = 0
        # Dynamics model.
        self.set_dynamics_func()
        # CasADi optimizer.
        self.setup_optimizer(self.solver)
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None

        self.setup_results_dict()

    def set_dynamics_func(self):
        '''Updates symbolic dynamics with actual control frequency.'''
        # linear dynamics for LQR ancillary gain and terminal cost
        dfdxdfdu = self.model.df_func(x=np.atleast_2d(self.model.X_EQ)[0, :].T,
                                      u=np.atleast_2d(self.model.U_EQ)[0, :].T)
        dfdx = dfdxdfdu['dfdx'].toarray()
        dfdu = dfdxdfdu['dfdu'].toarray()
        delta_x = cs.MX.sym('delta_x', self.model.nx, 1)
        delta_u = cs.MX.sym('delta_u', self.model.nu, 1)
        Ad, Bd = discretize_linear_system(dfdx, dfdu, self.dt, exact=True)
        x_dot_lin = Ad @ delta_x + Bd @ delta_u
        self.linear_dynamics_func = cs.Function('linear_discrete_dynamics',
                                                [delta_x, delta_u],
                                                [x_dot_lin],
                                                ['x0', 'p'],
                                                ['xf'])
        self.dfdx = dfdx
        self.dfdu = dfdu
        # # check controlled system is stabilizable
        # A = dfdx
        # B = dfdu
        # n = self.model.nx
        # m = self.model.nu
        # import control
        # ctrb = control.ctrb(A, B)
        # if np.linalg.matrix_rank(ctrb) != n:
        #     raise Exception('System is not stabilizable')
        try:
            self.lqr_gain, _, _, self.P = \
                compute_discrete_lqr_gain_from_cont_linear_system(dfdx,
                                                                  dfdu,
                                                                  self.Q,
                                                                  self.R,
                                                                  self.dt)
        except LinAlgError:
            print(colored('LQR gain computation failed', 'red'))
            print(colored('Using the LQR gain and terminal cost in the MPC is disabled', 'yellow'))
            self.use_lqr_gain_and_terminal_cost = False
            
        # nonlinear dynamics
        self.dynamics_func = rk_discrete(self.model.fc_func,
                                         self.model.nx,
                                         self.model.nu,
                                         self.dt)

    @timing
    def compute_initial_guess(self, init_state, goal_states=None):
        '''Compute an initial guess of the solution to the optimization problem.'''
        if goal_states is None:
            goal_states = self.get_references()
        print(colored(f'computing initial guess using {self.compute_initial_guess_method}', 'green'))
        if self.compute_initial_guess_method == 'ipopt':
            self.setup_optimizer(solver=self.init_solver)
            opti_dict = self.opti_dict
            opti = opti_dict['opti']
            x_var = opti_dict['x_var']  # optimization variables
            u_var = opti_dict['u_var']  # optimization variables
            x_init = opti_dict['x_init']  # initial state
            x_ref = opti_dict['x_ref']  # reference state/trajectory

            # Assign the initial state.
            opti.set_value(x_init, init_state)  # initial state should have dim (nx,)
            # Assign reference trajectory within horizon.
            opti.set_value(x_ref, goal_states)
            # Solve the optimization problem.
            try:
                sol = opti.solve()
                x_val, u_val = sol.value(x_var), sol.value(u_var)
            except RuntimeError:
                print(colored('Warm-starting fails', 'red'))
                x_val, u_val = opti.debug.value(x_var), opti.debug.value(u_var)
            x_guess = x_val
            u_guess = u_val
        elif self.compute_initial_guess_method == 'lqr':
            # initialize the guess solutions
            x_guess = np.zeros((self.model.nx, self.T + 1))
            u_guess = np.zeros((self.model.nu, self.T))
            x_guess[:, 0] = init_state
            # add the lqr gain and states to the guess
            for i in range(self.T):
                u = self.lqr_gain @ (x_guess[:, i] - goal_states[:, i]) + np.atleast_2d(self.model.U_EQ)[0, :].T
                u_guess[:, i] = u
                x_guess[:, i + 1, None] = self.dynamics_func(x0=x_guess[:, i], p=u)['xf'].toarray()
        else:
            raise Exception('Initial guess method not implemented.')

        self.x_prev = x_guess
        self.u_prev = u_guess

        # set the solver back
        self.setup_optimizer(solver=self.solver)

        return x_guess, u_guess

    def setup_optimizer(self, solver='qrsqp'):
        '''Sets up nonlinear optimization problem.'''
        print(colored(f'Setting up optimizer with {solver}', 'green'))
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        # Define optimizer and variables.
        opti = cs.Opti()
        # States.
        x_var = opti.variable(nx, T + 1)
        # Inputs.
        u_var = opti.variable(nu, T)
        # Initial state.
        x_init = opti.parameter(nx, 1)
        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = opti.parameter(nx, T + 1)
        # Add slack variables
        state_slack = opti.variable(len(self.state_constraints_sym))
        input_slack = opti.variable(len(self.input_constraints_sym))

        # cost (cumulative)
        cost = 0
        cost_func = self.model.loss
        for i in range(T):
            # Can ignore the first state cost since fist x_var == x_init.
            cost += cost_func(x=x_var[:, i],
                              u=u_var[:, i],
                              Xr=x_ref[:, i],
                              Ur=self.U_EQ,
                              Q=self.Q,
                              R=self.R)['l']
        # Terminal cost.
        cost += cost_func(x=x_var[:, -1],
                          u=np.zeros((nu, 1)),
                          Xr=x_ref[:, -1],
                          Ur=self.U_EQ,
                          Q=self.Q if not self.use_lqr_gain_and_terminal_cost else self.P,
                          R=self.R)['l']
        # Constraints
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.dynamics_func(x0=x_var[:, i], p=u_var[:, i])['xf']
            opti.subject_to(x_var[:, i + 1] == next_state)

            for sc_i, state_constraint in enumerate(self.state_constraints_sym):
                if self.soft_constraints:
                    opti.subject_to(state_constraint(x_var[:, i]) <= state_slack[sc_i])
                    cost += self.soft_penalty * state_slack[sc_i]**2
                    opti.subject_to(state_slack[sc_i] >= 0)
                else:
                    opti.subject_to(state_constraint(x_var[:, i]) < -self.constraint_tol)
            for ic_i, input_constraint in enumerate(self.input_constraints_sym):
                if self.soft_constraints:
                    opti.subject_to(input_constraint(u_var[:, i]) <= input_slack[ic_i])
                    cost += self.soft_penalty * input_slack[ic_i]**2
                    opti.subject_to(input_slack[ic_i] >= 0)
                else:
                    opti.subject_to(input_constraint(u_var[:, i]) < -self.constraint_tol)

        # Final state constraints.
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            if self.soft_constraints:
                opti.subject_to(state_constraint(x_var[:, -1]) <= state_slack[sc_i])
                cost += self.soft_penalty * state_slack[sc_i] ** 2
                opti.subject_to(state_slack[sc_i] >= 0)
            else:
                opti.subject_to(state_constraint(x_var[:, -1]) <= -self.constraint_tol)
        # initial condition constraints
        opti.subject_to(x_var[:, 0] == x_init)

        opti.minimize(cost)
        # Create solver
        opts = {'expand': True, 'error_on_fail': False}
        opti.solver(solver, opts)

        self.opti_dict = {
            'opti': opti,
            'x_var': x_var,
            'u_var': u_var,
            'x_init': x_init,
            'x_ref': x_ref,
            'cost': cost
        }

    @timing
    def select_action(self,
                      obs,
                      info=None
                      ):
        '''Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info

        Returns:
            action (ndarray): Input/action to the task/env.
        '''
        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        x_var = opti_dict['x_var']  # optimization variables
        u_var = opti_dict['u_var']  # optimization variables
        x_init = opti_dict['x_init']  # initial state
        x_ref = opti_dict['x_ref']  # reference state/trajectory

        # Assign the initial state.
        opti.set_value(x_init, obs)
        # Assign reference trajectory within horizon.
        goal_states = self.get_references()
        opti.set_value(x_ref, goal_states)

        if self.compute_initial_guess_method is not None and self.x_prev is None and self.u_prev is None:
            x_guess, u_guess = self.compute_initial_guess(obs)
            opti.set_initial(x_var, x_guess)
            opti.set_initial(u_var, u_guess)  # Initial guess for optimization problem.
        if self.warmstart and self.x_prev is not None and self.u_prev is not None:
            # shift previous solutions by 1 step
            x_guess = deepcopy(self.x_prev)
            u_guess = deepcopy(self.u_prev)
            x_guess[:, :-1] = x_guess[:, 1:]
            u_guess[:-1] = u_guess[1:]
            opti.set_initial(x_var, x_guess)
            opti.set_initial(u_var, u_guess)

        if self.mode == 'tracking':
            # increment the trajectory step after update the reference and initial guess
            self.traj_step += 1

        # Solve the optimization problem.
        try:
            sol = opti.solve()
            x_val, u_val = sol.value(x_var), sol.value(u_var)
        except RuntimeError:
            print(colored('Infeasible MPC Problem', 'red'))
            return_status = opti.return_status()
            print(colored(f'Optimization failed with status: {return_status}', 'red'))
            if self.solver == 'ipopt':
                x_val, u_val = opti.debug.value(x_var), opti.debug.value(u_var)
            elif self.solver == 'qrsqp':
                if return_status == 'unknown':
                    # self.terminate_loop = True
                    if self.u_prev is None:
                        print(colored('[WARN]: MPC Infeasible first step.', 'yellow'))
                        u_val = np.zeros((self.model.nu, self.T))
                        x_val = np.zeros((self.model.nx, self.T + 1))
                    else:
                        u_val = self.u_prev
                        x_val = self.x_prev
                elif return_status in ['Maximum_Iterations_Exceeded', 'Infeasible_Problem_Detected']:
                    self.terminate_loop = True
                    u_val = opti.debug.value(u_var)
                    x_val = opti.debug.value(x_var)
        self.x_prev = x_val
        self.u_prev = u_val
        self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
        self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
        self.results_dict['goal_states'].append(deepcopy(goal_states))
        if self.solver == 'ipopt':
            self.results_dict['t_wall'].append(opti.stats()['t_wall_total'])
        # Take the first action from the solved action sequence.
        if u_val.ndim > 1:
            action = u_val[:, 0]
        else:
            action = np.array([u_val[0]])
        if self.use_lqr_gain_and_terminal_cost:
            action += self.lqr_gain @ (obs - x_val[:, 0])
        self.prev_action = action
        return action
    @timing
    def get_references(self):
        '''Constructs reference states along mpc horizon.(nx, T+1).'''

        # if the task is to track a periodic trajectory (circle, square, figure 8)
        # append the T+1 states of the trajectory to the goal_states 
        # such that the vel states won't drop at the end of an episode
        if self.env.TASK == Task.STABILIZATION:
            # Repeat goal state for horizon steps.
            goal_states = np.tile(self.env.X_GOAL.reshape(-1, 1), (1, self.T + 1))
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.extended_ref_traj = deepcopy(self.traj)
            if self.env.TASK_INFO['trajectory_type'] in ['circle', 'square', 'figure8']:
                self.extended_ref_traj = np.concatenate([self.extended_ref_traj, self.extended_ref_traj[:, :self.T+1]], axis=1)
            # Slice trajectory for horizon steps, if not long enough, repeat last state.
            start = min(self.traj_step, self.extended_ref_traj.shape[-1])
            end = min(self.traj_step + self.T + 1, self.extended_ref_traj.shape[-1])
            remain = max(0, self.T + 1 - (end - start)) 
            '''
            TODO: if using the extended reference trajectory, 
            variable remain will always be 0. Consider removing it.
            '''
            print('start:', start, 'end:', end, 'remain:', remain)
            goal_states = np.concatenate([
                self.extended_ref_traj[:, start:end],
                np.tile(self.extended_ref_traj[:, -1:], (1, remain))
            ], -1)
        else:
            raise Exception('Reference for this mode is not implemented.')
        return goal_states  # (nx, T+1).

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information.'''
        self.results_dict = {'obs': [],
                             'reward': [],
                             'done': [],
                             'info': [],
                             'action': [],
                             'horizon_inputs': [],
                             'horizon_states': [],
                             'goal_states': [],
                             'frames': [],
                             'state_mse': [],
                             'common_cost': [],
                             'state': [],
                             'state_error': [],
                             't_wall': []
                             }

    def run(self,
            env=None,
            render=False,
            logging=False,
            max_steps=None,
            terminate_run_on_done=None
            ):
        '''Runs evaluation with current policy.

        Args:
            render (bool): if to do real-time rendering.
            logging (bool): if to log on terminal.

        Returns:
            dict: evaluation statisitcs, rendered frames.
        '''
        if env is None:
            env = self.env  
        if terminate_run_on_done is None:
            terminate_run_on_done = self.terminate_run_on_done

        self.x_prev = None
        self.u_prev = None
        if not env.initial_reset:
            env.set_cost_function_param(self.Q, self.R)
        obs, info = env.reset()
        # obs = env.reset()
        print('Init State:')
        print(obs)
        ep_returns, ep_lengths = [], []
        frames = []
        self.setup_results_dict()
        self.results_dict['obs'].append(obs)
        self.results_dict['state'].append(env.state)
        i = 0
        if env.TASK == Task.STABILIZATION:
            if max_steps is None:
                MAX_STEPS = int(env.CTRL_FREQ * env.EPISODE_LEN_SEC)
            else:
                MAX_STEPS = max_steps
        elif env.TASK == Task.TRAJ_TRACKING:
            if max_steps is None:
                MAX_STEPS = self.traj.shape[1]
            else:
                MAX_STEPS = max_steps
        else:
            raise Exception('Undefined Task')
        self.terminate_loop = False
        done = False
        common_metric = 0
        while not (done and terminate_run_on_done) and i < MAX_STEPS and not (self.terminate_loop):
            action = self.select_action(obs)
            if self.terminate_loop:
                print('Infeasible MPC Problem')
                break
            # Repeat input for more efficient control.
            obs, reward, done, info = env.step(action)
            self.results_dict['obs'].append(obs)
            self.results_dict['reward'].append(reward)
            self.results_dict['done'].append(done)
            self.results_dict['info'].append(info)
            self.results_dict['action'].append(action)
            self.results_dict['state'].append(env.state)
            self.results_dict['state_mse'].append(info['mse'])
            self.results_dict['state_error'].append(env.state - env.X_GOAL[i, :])
            common_metric += info['mse']
            print(i, '-th step.')
            print('action:', action)
            print('obs', obs)
            print('reward', reward)
            print('done', done)
            print(info)
            print()
            if render:
                env.render()
                frames.append(env.render('rgb_array'))
            i += 1
        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        if logging:
            msg = '****** Evaluation ******\n'
            msg += 'eval_ep_length {:.2f} +/- {:.2f} | eval_ep_return {:.3f} +/- {:.3f}\n'.format(
                ep_lengths.mean(), ep_lengths.std(), ep_returns.mean(),
                ep_returns.std())
        if len(frames) != 0:
            self.results_dict['frames'] = frames
        self.results_dict['obs'] = np.vstack(self.results_dict['obs'])
        self.results_dict['state'] = np.vstack(self.results_dict['state'])
        try:
            self.results_dict['reward'] = np.vstack(self.results_dict['reward'])
            self.results_dict['action'] = np.vstack(self.results_dict['action'])
            self.results_dict['full_traj_common_cost'] = common_metric
            self.results_dict['total_rmse_state_error'] = compute_state_rmse(self.results_dict['state'])
            self.results_dict['total_rmse_obs_error'] = compute_state_rmse(self.results_dict['obs'])
        except ValueError:
            raise Exception('[ERROR] mpc.run().py: MPC could not find a solution for the first step given the initial conditions. '
                            'Check to make sure initial conditions are feasible.')
        return deepcopy(self.results_dict)

    def reset_before_run(self, obs, info=None, env=None):
        '''Reinitialize just the controller before a new run.

        Args:
            obs (ndarray): The initial observation for the new run.
            info (dict): The first info of the new run.
            env (BenchmarkEnv): The environment to be used for the new run.
        '''
        self.reset()
