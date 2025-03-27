'''Flatness based Model Predictive Control
for 2D attitude Quadrotor, with system extension

Based on:
    Linear MPC 
'''


from copy import deepcopy
from sys import platform

import casadi as cs
import numpy as np

from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.controllers.mpc.linear_mpc import LinearMPC
# from safe_control_gym.controllers.mpc.linear_mpc_acados import LinearMPC_ACADOS
from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.mpc.mpc_utils import get_cost_weight_matrix
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel

class FlatMPC_EXT(BaseController):
    '''Flatness based MPC for system with dynamic extension.'''

    def __init__(
            self,
            env_func,
            horizon=5,
            q_mpc=[1],
            r_mpc=[1],
            warmstart=True,
            soft_constraints=False,
            terminate_run_on_done=True,
            constraint_tol: float = 1e-8,
            solver: str = 'sqpmethod',
            use_full_flat_reference=True,
            output_dir='results/temp',
            additional_constraints=None,
            use_acados=False,
            flat_state_constraint = dict,
            extended_input_clipping = dict,
            **kwargs):
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
            solver (str): Specify which solver you wish to use (qrqp, qpoases, ipopt, sqpmethod)
            use_full_flat_reference (bool): Use reference with acceleration and jerk for figure8 and circle trajectories
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): list of constraints.
            use_acados: use acados linear MPC or CasADi linear MPC
            flat_state_constraint: dict with parameters for flat half space constraint
            extended_input_clipping: dict with limits to clip extended input with
        '''
        # Store all params/args.
        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__[k] = v

        if use_acados:
            self.mpc = LinearMPC_ACADOS(
                                        env_func,
                                        horizon=horizon,
                                        q_mpc=[1],
                                        r_mpc=[1],
                                        warmstart=warmstart,
                                        soft_constraints=soft_constraints,
                                        terminate_run_on_done=terminate_run_on_done,
                                        constraint_tol=constraint_tol,
                                        output_dir=output_dir,
                                        additional_constraints=additional_constraints,
                                        **kwargs
                                    )
        else:
            self.mpc = LinearMPC(
                            env_func,
                            horizon=horizon,
                            q_mpc=[1],
                            r_mpc=[1],
                            warmstart=warmstart,
                            soft_constraints=soft_constraints,
                            terminate_run_on_done=terminate_run_on_done,
                            constraint_tol=constraint_tol,
                            output_dir=output_dir,
                            additional_constraints=additional_constraints,
                            **kwargs
                        )
                        
        self.env = self.mpc.env # just for debugging, not used in code below

        self.QUAD_TYPE = self.mpc.env.QUAD_TYPE
        if self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
            raise NotImplementedError("fmpc_ext only implemented for 2D attitude model")
        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
            self.transform_env_goal_to_flat_func = _transform_env_goal_to_flat_2D_att 
            # replace dynamics model with symbolic flat model
            self.mpc.model = _setup_flat_model_symbolic_2D_att(self.mpc.dt)
            self.inertial_prop = self.mpc.env.INERTIAL_PROP
        else:
            raise NotImplementedError("fmpc_ext only implemented for 2D attitude model")    
        
        self.use_full_flat_reference = use_full_flat_reference 

        if use_acados:
            self.mpc.x_lin = np.atleast_2d(self.mpc.model.X_EQ)[0, :].T
            self.mpc.u_lin = np.atleast_2d(self.mpc.model.U_EQ)[0, :].T 
        else:
            self.mpc.X_EQ = np.atleast_2d(self.mpc.model.X_EQ)[0, :].T
            self.mpc.U_EQ = np.atleast_2d(self.mpc.model.U_EQ)[0, :].T 
        assert solver in ['qpoases', 'qrqp', 'sqpmethod', 'ipopt'], '[Error]. MPC Solver not supported.'
        self.mpc.solver = solver

        # overwrite definitions in parent init function to fit flat model
        self.mpc.Q = get_cost_weight_matrix(q_mpc, self.mpc.model.nx) 
        self.mpc.R = get_cost_weight_matrix(r_mpc, self.mpc.model.nu)
        
        # remove all constraints from MPC
        self.mpc.constraints = []
        self.mpc.state_constraints_sym = []
        self.mpc.input_constraints_sym = [] 

        # adding half space constraint on flat state
        if flat_state_constraint.apply_state_bound == True:
            h = np.atleast_2d(np.array(flat_state_constraint.h_vect)).T
            assert np.shape(h)[0] == self.mpc.model.nx, "Flat half space constraint: dimension of h does not fit flat state dim"
            b = flat_state_constraint.b_val
            sym_func = lambda x: h.T @ x - b
            self.mpc.state_constraints_sym = [sym_func]

        # setup flat state observer
        self.fs_obs = FlatStateObserver(self.QUAD_TYPE, self.inertial_prop, self.mpc.env.GRAVITY_ACC, self.mpc.dt, self.mpc.T)

        # setup double integrator for dynamic extension
        self.eta = np.zeros(2)
        A_dyn_ext = np.zeros((2, 2))
        A_dyn_ext[0, 1] = 1.0
        B_dyn_ext = np.zeros((2, 2))
        B_dyn_ext[1, 0] = 1.0
        self.Ad_dyn_ext, self.Bd_dyn_ext = discretize_linear_system(A_dyn_ext, B_dyn_ext, self.mpc.dt, exact=True)

        # setup input clipping
        self.extended_input_clipping = extended_input_clipping.apply_input_clipping
        if self.extended_input_clipping:
            self.input_limit = np.array(extended_input_clipping.input_limit)

    def reset(self):
        '''Prepares for training or evaluation.'''
        self.mpc.reset()
        # Setup reference input for the flat state spaces
        if self.mpc.env.TASK == Task.STABILIZATION:
            self.mpc.mode = 'stabilization'
            self.mpc.x_goal = self.transform_env_goal_to_flat_func(self.mpc.env.X_GOAL)            
            x_ini = self.env.__dict__['init_x'.upper()]
            y_ini = self.env.__dict__.get('init_y'.upper(), 0)
            z_ini = self.env.__dict__['init_z'.upper()]
            self.fs_obs.set_initial_hovering(x_ini, y_ini, z_ini)
        elif self.mpc.env.TASK == Task.TRAJ_TRACKING:
            self.mpc.mode = 'tracking'
            if self.use_full_flat_reference:
                self.mpc.traj = get_full_reference_trajectory_FMPC(self.QUAD_TYPE, self.mpc.env.TASK_INFO, self.mpc.env.EPISODE_LEN_SEC, self.mpc.dt, self.mpc.T).T
            else:
                self.mpc.traj = self.transform_env_goal_to_flat_func(self.mpc.env.X_GOAL.T)            
            # Step along the reference.
            self.mpc.traj_step = 0
            # initialize flat state observer in hovering
            x_ini = self.mpc.env.__dict__['init_x'.upper()]
            y_ini = self.mpc.env.__dict__.get('init_y'.upper(), 0)
            z_ini = self.mpc.env.__dict__['init_z'.upper()]
            self.fs_obs.set_initial_hovering(x_ini, y_ini, z_ini)

         # initialize dynamic extension in hovering
        if self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
            raise NotImplementedError
        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
            self.eta[0] = (self.mpc.env.GRAVITY_ACC- self.inertial_prop['beta_2'])/self.inertial_prop['beta_1']
        
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
                             't_wall': [],
                            'z_inp':[],
                            'v_inp':[],
                            'u_ext':[]
                             }

    def select_action(self,
                      obs,
                      info=None
                      ):
        '''Solve nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info.

        Returns:
            action (ndarray): Input/action to the task/env.
        '''  
        # get flat state estimation from observer
        z_obs = self.fs_obs.compute_observation(obs)
    
        # run MPC controller 
        v = self.mpc.select_action(z_obs) 
        z_horizon = self.mpc.x_prev #8xN set in linearMPC
        v_horizon = self.mpc.u_prev #2xN       
        
        # flat input transformation: z and v to action u        
        zd = z_horizon[:, 0]
        vd = v_horizon[:, 0]
        action_extended = _get_u_from_flat_states_2D_att_ext(zd, vd, self.inertial_prop, self.mpc.env.GRAVITY_ACC) # for dynamics with system extension
        if self.extended_input_clipping:
            action_extended = np.clip(action_extended, -self.input_limit, self.input_limit)
       
        # do double integration on first action Tc_ddot --> Tc
        self.eta = self.Ad_dyn_ext @ self.eta + self.Bd_dyn_ext @ action_extended
        action = np.zeros(np.shape(action_extended))
        action[0] = self.eta[0]
        action[1] = action_extended[1]

        # feed data into observer
        self.fs_obs.input_FMPC_result(z_horizon, v_horizon, action)

        # data logging
        self.results_dict['z_inp'].append(zd)
        self.results_dict['v_inp'].append(vd)
        self.results_dict['u_ext'].append(action_extended)
        
        return action
    
    def close(self):
        '''Cleans up resources.'''
        self.mpc.close()
    
class FlatStateObserver():
    
    def __init__(self,  QUAD_TYPE: QuadType, inertial_prop, g:float, dt: float, horizon:int):
        '''Creates observer for flat state model

        Args:
            QUAD_TYPE (QuadType): Quadrotor type from enviroment (2D/3D, attitude model, etc.)
            inertial_prop:        Inertial properties of the quadrotor model, other identified parameters, from env or config file
            g : gravity acceleration constant
            dt: time step size, 1/control frequency
            horizon: FMPC horizon length
        '''
        self.QUAD_TYPE = QUAD_TYPE
        self.inertial_prop = inertial_prop 
        self.GRAVITY = g
        self.dt = dt
        self.fmpc_horizon = horizon
        
        if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
            self.action_from_flat_states_func = _get_u_from_flat_states_2D_att
            self.flat_states_from_reg_func = _get_z_from_regular_states_2D_att
        else:
            raise NotImplementedError('FMPC flat state observer only implemented for 2D_attitude model')


    def set_initial_hovering(self, x_pos, y_pos, z_pos):       
        if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
            self.z_horizon = np.zeros([8, self.fmpc_horizon+1])
            self.v_horizon = np.zeros([2, self.fmpc_horizon])
            self.u = np.zeros(2)
            self.u[0] = (self.GRAVITY- self.inertial_prop['beta_2'])/self.inertial_prop['beta_1']
            z_ini = np.zeros(8)
            z_ini[0] = x_pos
            z_ini[4] = z_pos           
        else: 
            raise NotImplementedError('FMPC flat state observer initial hovering only implemented for 2D_attitude model')
       
        self.z_horizon = np.tile(z_ini.reshape(-1, 1), (1, self.fmpc_horizon + 1)) 
       
    def input_FMPC_result(self, z_horizon, v_horizon, u):
        # just save them away
        self.z_horizon = z_horizon
        self.v_horizon = v_horizon
        self.u = u     
              
    def compute_observation(self, x_obs): 
        # estimate u_dot at current time step, based on z_horizon and v_horizon set in last time step
        u_comp_length = 3
        if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
            u_horizon = np.zeros([2, u_comp_length])
        else:
            raise NotImplementedError('FMPC flat state observer compute observation only implemented for 2D_attitude model')
        
        for i in range(u_comp_length):
            u_horizon[:, i] = self.action_from_flat_states_func(self.z_horizon[:,i], self.v_horizon[:,i], self.inertial_prop, self.GRAVITY)

        u_dot_central = (-u_horizon[:, 0]  + u_horizon[:, 2])/(2*self.dt)
       
        u0_dot = u_dot_central[0]

        # state estimation using system dynamics
        z_obs = self.flat_states_from_reg_func(x_obs, self.u[0], u0_dot, self.inertial_prop, self.GRAVITY) 
        return z_obs
    

#################################################################################################
################## 2D Quadrotor SI model flatness transforms ####################################
################################################################################################# 

def _setup_flat_model_symbolic_2D_att(dt):
    '''Generates linear flat model for 2D SI model
    Integrator chain for x y and z 

    Args:
        dt: time step size of controller, 1/control frequency
    ''' 
    nx, nu = 8, 2
    
    # Define states.
    z = cs.MX.sym('z')
    z_dot = cs.MX.sym('z_dot')
    z_ddot = cs.MX.sym('z_ddot')   
    z_dddot = cs.MX.sym('z_dddot')       
    
    x = cs.MX.sym('x')
    x_dot = cs.MX.sym('x_dot')
    x_ddot = cs.MX.sym('x_ddot')
    x_dddot = cs.MX.sym('x_dddot')
    
    X = cs.vertcat(x, x_dot, x_ddot, x_dddot, z, z_dot, z_ddot, z_dddot)
    # Define flat inputs 
    v1 = cs.MX.sym('v1')
    v2 = cs.MX.sym('v2')
    U = cs.vertcat(v1, v2)
    # Define dynamics equations.
    X_dot = cs.vertcat(x_dot, x_ddot, x_dddot, v1,
                        z_dot, z_ddot, z_dddot, v2)
    # Define observation.
    Y = cs.vertcat(x, x_dot, x_ddot, x_dddot, z, z_dot, z_ddot, z_dddot)
    
    # Set the equilibrium values for linearizations.
    X_EQ = np.zeros(nx)
    U_EQ = np.zeros(nu)
    # Define cost (quadratic form).
    Q = cs.MX.sym('Q', nx, nx)
    R = cs.MX.sym('R', nu, nu)
    Xr = cs.MX.sym('Xr', nx, 1)
    Ur = cs.MX.sym('Ur', nu, 1)
    cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
    # Define dynamics and cost dictionaries.
    dynamics = {'dyn_eqn': X_dot, 'obs_eqn': Y, 'vars': {'X': X, 'U': U}}
    cost = {
        'cost_func': cost_func,
        'vars': {
            'X': X,
            'U': U,
            'Xr': Xr,
            'Ur': Ur,
            'Q': Q,
            'R': R
        }
    }
    # Additional params to cache
    params = {
        # equilibrium point for linearization
        'X_EQ': X_EQ,
        'U_EQ': U_EQ,
    }
    # Setup symbolic model.
    return SymbolicModel(dynamics=dynamics, cost=cost, dt=dt, params=params)

def _get_u_from_flat_states_2D_att(z, v, dyn_pars, g):
    beta_1 = dyn_pars['beta_1']
    beta_2 = dyn_pars['beta_2']
    alpha_1 =  dyn_pars['alpha_1']
    alpha_2 =  dyn_pars['alpha_2']
    alpha_3 =  dyn_pars['alpha_3']

    term_acc_sqrd = (z[2])**2 + (z[6]+g)**2 # x_ddot^2 + (z_ddot+g)^2
    theta = np.arctan2(z[2], (z[6]+g))
    theta_dot = (z[3]*(z[6]+g)- z[2]*z[7])/term_acc_sqrd
    theta_ddot = 1/term_acc_sqrd * (v[0]*(z[6]+g) - z[2]*v[1]) + (1/(term_acc_sqrd**2)) * (2*(z[6]+g)*z[7] + 2*z[2]*z[3]) * (z[2]*z[7] - z[3]*(z[6]+g))

    t = -(beta_2/beta_1) + np.sqrt(term_acc_sqrd)/beta_1
    p = (1/alpha_3) * (theta_ddot - alpha_1*theta -alpha_2*theta_dot)
    return np.array([t, p])

# for system with dynamic extension
def _get_u_from_flat_states_2D_att_ext(z, v, dyn_pars, g):
    # for system with dynamic extension: u = [Tc_ddot, theta_c]
    beta_1 = dyn_pars['beta_1']
    beta_2 = dyn_pars['beta_2']
    alpha_1 =  dyn_pars['alpha_1']
    alpha_2 =  dyn_pars['alpha_2']
    alpha_3 =  dyn_pars['alpha_3']

    term_acc_sqrd = (z[2])**2 + (z[6]+g)**2 # x_ddot^2 + (z_ddot+g)^2
    theta = np.arctan2(z[2], (z[6]+g))
    theta_dot = (z[3]*(z[6]+g)- z[2]*z[7])/term_acc_sqrd
    theta_ddot = 1/term_acc_sqrd * (v[0]*(z[6]+g) - z[2]*v[1]) + (1/(term_acc_sqrd**2)) * (2*(z[6]+g)*z[7] + 2*z[2]*z[3]) * (z[2]*z[7] - z[3]*(z[6]+g))

    #t = -(beta_2/beta_1) + np.sqrt(term_acc_sqrd)/beta_1
    p = (1/alpha_3) * (theta_ddot - alpha_1*theta -alpha_2*theta_dot)

    t_ddot = 1/beta_1 * 1/np.sqrt(term_acc_sqrd)*((z[3]**2 + z[7]**2 + z[2]*v[0] + (z[6]+g)*v[1]) - ((z[2]*z[3] + (z[6]+g)*z[7])**2)/term_acc_sqrd)
    return np.array([t_ddot, p])

def _get_z_from_regular_states_2D_att(x, u0, u0_dot, dyn_pars, g):    
   
    beta_1 = dyn_pars['beta_1']
    beta_2 = dyn_pars['beta_2']        
       
    z = np.zeros(8)
    sin_theta = np.sin(x[4])
    cos_theta = np.cos(x[4])

    z[0] = x[0] # x
    z[1] = x[1] # x_dot
    z[2] = sin_theta*(beta_2 + beta_1*u0) # x_ddot    
    z[3] = cos_theta*(beta_2 + beta_1*u0)*x[5] + sin_theta*beta_1*u0_dot # x_dddot
    z[4] = x[2] # z
    z[5] = x[3] # z_dot
    z[6] = cos_theta*(beta_2 + beta_1*u0)- g # z_ddot
    z[7] = -sin_theta*(beta_2 + beta_1*u0)*x[5] + cos_theta*beta_1*u0_dot# z_dddot
    return z

# not needed in FMPC, used to double check transformations
def _get_x_from_flat_states_2D_att(z, g):
    x = np.zeros(6)
    x[0] = z[0]
    x[1] = z[1]
    x[2] = z[4]
    x[3] = z[5]
    x[4] = np.arctan2(z[2], (z[6]+g))
    x[5] = (z[3]*(z[6]+g)- z[2]*z[7])/((z[6]+g)**2 + z[2]**2)
    return x

def _transform_env_goal_to_flat_2D_att(x):
    if x.ndim == 1:
        l = 1
    else:
        l = np.shape(x)[1]
    z = np.zeros((8, l) )
    z[0, ...] = x[0, ...]
    z[1, ...] = x[1, ...]
    z[4, ...] = x[2, ...]
    z[5, ...] = x[3, ...]
    return z


#################################################################################################
###################### Trajectory generation ####################################################
################################################################################################# 

def get_full_reference_trajectory_FMPC(QUAD_TYPE: QuadType, 
                                task_info, 
                                traj_length,
                                sample_time=0.01, 
                                horizon=0):
    """Generates a 2D trajectory with acceleration and jerk for a full flat reference

    Args:
        QUAD_TYPE: QuadType object specifying 2D/3D quad model
        task_info: task information from environment (scale, cycles, offset, plane, type)
        traj_legth: time duration of the whole trajectory
        sample_time: time step size of reference, 1/controller_freq
        horizon: FMPC horizon length in timesteps, to extend trajectory accordingly

    Returns:
        ndarray: array with full reference on flat state vector
    """
    
    # task info parameters from yaml file
    scaling = task_info.trajectory_scale
    num_cycles = task_info.num_cycles
    position_offset = task_info.trajectory_position_offset
    traj_plane = task_info.trajectory_plane
    traj_type = task_info.trajectory_type

    pos_ref_traj, vel_ref_traj, acc_ref_traj, jer_ref_traj = _generate_trajectory_FMPC(traj_type, traj_length, num_cycles, traj_plane, position_offset, scaling, sample_time, horizon)
    num_times = np.shape(pos_ref_traj)[0]
    if QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
        z_ref = np.zeros([num_times, 12])
        z_ref[:,0] = pos_ref_traj[:, 0]
        z_ref[:,1] = vel_ref_traj[:, 0]
        z_ref[:,2] = acc_ref_traj[:, 0]
        z_ref[:,3] = jer_ref_traj[:, 0]
        z_ref[:,4] = pos_ref_traj[:, 1]
        z_ref[:,5] = vel_ref_traj[:, 1]
        z_ref[:,6] = acc_ref_traj[:, 1]
        z_ref[:,7] = jer_ref_traj[:, 1]
        z_ref[:,8] = pos_ref_traj[:, 2] 
        z_ref[:,9] = vel_ref_traj[:, 2]
        z_ref[:,10] = acc_ref_traj[:, 2]
        z_ref[:,11] = jer_ref_traj[:, 2]
    elif QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
        z_ref = np.zeros([num_times, 8])
        z_ref[:,0] = pos_ref_traj[:, 0]
        z_ref[:,1] = vel_ref_traj[:, 0]
        z_ref[:,2] = acc_ref_traj[:, 0]
        z_ref[:,3] = jer_ref_traj[:, 0]
        z_ref[:,4] = pos_ref_traj[:, 2] 
        z_ref[:,5] = vel_ref_traj[:, 2]
        z_ref[:,6] = acc_ref_traj[:, 2]
        z_ref[:,7] = jer_ref_traj[:, 2]
    else:
        raise NotImplementedError('Flat reference not implemented for this quadrotor type, only for 2D_attitude and 3D_attitude_10')
    
    return z_ref



def _generate_trajectory_FMPC(traj_type='figure8',
                             traj_length=10.0,
                             num_cycles=1,
                             traj_plane='xy',
                             position_offset=np.array([0, 0]),
                             scaling=1.0,
                             sample_time=0.01, 
                             horizon=0
                             ):
    """Generates a 2D trajectory.

    Args:
        traj_type (str, optional): The type of trajectory (circle, square, figure8).
        traj_length (float, optional): The length of the trajectory in seconds.
        num_cycles (int, optional): The number of cycles within the length.
        traj_plane (str, optional): The plane of the trajectory (e.g. 'xz').
        position_offset (ndarray, optional): An initial position offset in the plane.
        scaling (float, optional): Scaling factor for the trajectory.
        sample_time (float, optional): The sampling timestep of the trajectory.
        horizon(int, optional): FMPC horizon, trajectory gets extended such that at the final timestep there still is a full horizon in the reference

    Returns:
        ndarray: The positions in x, y, z of the trajectory sampled for its entire duration.
        ndarray: The velocities in x, y, z of the trajectory sampled for its entire duration.
        ndarray: The acceleration in x, y, z of the trajectory sampled for its entire duration.
        ndarray: The jerk in x, y, z of the trajectory sampled for its entire duration.
    """

    # Get trajectory type.
    valid_traj_type = ['circle', 'figure8']
    if traj_type not in valid_traj_type:
        raise ValueError(
            'Trajectory type should be one of [circle, figure8] for FMPC full reference'
        )
    traj_period = traj_length / num_cycles
    direction_list = ['x', 'y', 'z']
    # Get coordinates indexes.
    if traj_plane[0] in direction_list and traj_plane[1] in direction_list and traj_plane[0] != traj_plane[1]:
        coord_index_a = direction_list.index(traj_plane[0])
        coord_index_b = direction_list.index(traj_plane[1])
    else:
        raise ValueError('Trajectory plane should be in form of ab, where a and b can be {x, y, z}.')
    # Generate time stamps.
    times = np.arange(0, traj_length + sample_time*(1+horizon), sample_time)  # sample time added to make reference one step longer than traj_length
    pos_ref_traj = np.zeros((len(times), 3))
    vel_ref_traj = np.zeros((len(times), 3))
    acc_ref_traj = np.zeros((len(times), 3))
    jer_ref_traj = np.zeros((len(times), 3))

    # Compute trajectory points.
    for t in enumerate(times):
        pos_ref_traj[t[0]], vel_ref_traj[t[0]], acc_ref_traj[t[0]], jer_ref_traj[t[0]] = _get_coordinates(t[1],
                                                                        traj_type,
                                                                        traj_period,
                                                                        coord_index_a,
                                                                        coord_index_b,
                                                                        position_offset[0],
                                                                        position_offset[1],
                                                                        scaling)
    # manually shift the z axis to 1.0 if not in the traj plane
    # otherwise flying on the floor with z=0.0 
    if 'z' not in traj_plane:
        pos_ref_traj[:, 2] = 1.0
        vel_ref_traj[:, 2] = 0.0
        
    return pos_ref_traj, vel_ref_traj, acc_ref_traj, jer_ref_traj

def _get_coordinates(t,
                    traj_type,
                    traj_period,
                    coord_index_a,
                    coord_index_b,
                    position_offset_a,
                    position_offset_b,
                    scaling
                    ):
    """Computes the coordinates of a specified trajectory at time t.

    Args:
        t (float): The time at which we want to sample one trajectory point.
        traj_type (str, optional): The type of trajectory (circle, figure8).
        traj_period (float): The period of the trajectory in seconds.
        coord_index_a (int): The index of the first coordinate of the trajectory plane.
        coord_index_b (int): The index of the second coordinate of the trajectory plane.
        position_offset_a (float): The offset in the first coordinate of the trajectory plane.
        position_offset_b (float): The offset in the second coordinate of the trajectory plane.
        scaling (float, optional): Scaling factor for the trajectory.

    Returns:
        pos_ref (ndarray): The position in x, y, z, at time t.
        vel_ref (ndarray): The velocity in x, y, z, at time t.
        acc_ref (ndarray): The velocity in x, y, z, at time t.
        jer_ref (ndarray): The velocity in x, y, z, at time t.
    """

    # Get coordinates for the trajectory chosen.
    if traj_type == 'figure8':
        coords_a, coords_b, coords_a_dot, coords_b_dot, coords_a_ddot, coords_b_ddot, coords_a_dddot, coords_b_dddot = _figure8(
            t, traj_period, scaling)
    elif traj_type == 'circle':
        coords_a, coords_b, coords_a_dot, coords_b_dot, coords_a_ddot, coords_b_ddot , coords_a_dddot, coords_b_dddot= _circle(
            t, traj_period, scaling)
    elif traj_type == 'square':
        raise NotImplementedError('Square reference not implemented in FMPC full reference generation')
    elif traj_type == 'snap_figure8':
        raise NotImplementedError('Snap_figure8 not implemented in FMPC full reference generation')
    # Initialize position and velocity references.
    pos_ref = np.zeros((3,))
    vel_ref = np.zeros((3,))
    acc_ref = np.zeros((3,))
    jer_ref = np.zeros((3,))
    # Set position and velocity references based on the plane of the trajectory chosen.
    pos_ref[coord_index_a] = coords_a + position_offset_a
    vel_ref[coord_index_a] = coords_a_dot
    acc_ref[coord_index_a] = coords_a_ddot
    jer_ref[coord_index_a] = coords_a_dddot
    pos_ref[coord_index_b] = coords_b + position_offset_b
    vel_ref[coord_index_b] = coords_b_dot
    acc_ref[coord_index_b] = coords_b_ddot
    jer_ref[coord_index_b] = coords_b_dddot
    return pos_ref, vel_ref, acc_ref, jer_ref

def _figure8(t,
                traj_period,
                scaling
                ):
    """Computes the coordinates of a figure8 trajectory at time t.

    Args:
        t (float): The time at which we want to sample one trajectory point.
        traj_period (float): The period of the trajectory in seconds.
        scaling (float, optional): Scaling factor for the trajectory.

    Returns:
        coords_a (float): The position in the first coordinate.
        coords_b (float): The position in the second coordinate.
        coords_a_dot (float): The velocity in the first coordinate.
        coords_b_dot (float): The velocity in the second coordinate.
    """
    traj_freq = 2.0 * np.pi / traj_period
    coords_a = scaling * np.sin(traj_freq * t)
    coords_a_dot = scaling * traj_freq * np.cos(traj_freq * t)
    coords_a_ddot = -scaling * traj_freq**2 * np.sin(traj_freq * t)
    coords_a_dddot = -scaling * traj_freq**3 * np.cos(traj_freq * t)
    coords_b = scaling * np.sin(traj_freq * t) * np.cos(traj_freq * t)        
    coords_b_dot = scaling * traj_freq * (np.cos(traj_freq * t)**2 - np.sin(traj_freq * t)**2)
    coords_b_ddot =  -scaling * traj_freq**2 * 4 * np.sin(traj_freq * t) *np.cos(traj_freq * t) 
    coords_b_dddot = scaling * traj_freq**3 * 4 * (np.sin(traj_freq * t)**2 - np.cos(traj_freq*t)**2)       

    return coords_a, coords_b, coords_a_dot, coords_b_dot, coords_a_ddot, coords_b_ddot, coords_a_dddot, coords_b_dddot


def _circle(t,
            traj_period,
            scaling
            ):
    """Computes the coordinates of a circle trajectory at time t.

    Args:
        t (float): The time at which we want to sample one trajectory point.
        traj_period (float): The period of the trajectory in seconds.
        scaling (float, optional): Scaling factor for the trajectory.

    Returns:
        coords_a (float): The position in the first coordinate.
        coords_b (float): The position in the second coordinate.
        coords_a_dot (float): The velocity in the first coordinate.
        coords_b_dot (float): The velocity in the second coordinate.
    """
    
    traj_freq = 2.0 * np.pi / traj_period
    coords_a = scaling * np.cos(traj_freq * t)
    coords_a_dot = -scaling * traj_freq * np.sin(traj_freq * t)
    coords_a_ddot = -scaling * traj_freq**2 * np.cos(traj_freq * t)
    coords_a_dddot = scaling * traj_freq**3 * np.sin(traj_freq * t)
    coords_b = scaling * np.sin(traj_freq * t)        
    coords_b_dot = scaling * traj_freq * np.cos(traj_freq * t)
    coords_b_ddot = -scaling * traj_freq**2 * np.sin(traj_freq * t)
    coords_b_dddot = -scaling * traj_freq**3 * np.cos(traj_freq * t)
    return coords_a, coords_b, coords_a_dot, coords_b_dot, coords_a_ddot, coords_b_ddot, coords_a_dddot, coords_b_dddot