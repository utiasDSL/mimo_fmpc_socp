import numpy as np
from numpy.linalg import norm
import cvxpy as cp
import torch

# for testing on its own
import os
import pickle
import gpytorch
from safe_control_gym.controllers.mpc.flat_gp_utils import ZeroMeanAffineGP, GaussianProcess
import matplotlib.pyplot as plt


# Make @profile decorator optional (only available when running with kernprof)
try:
    profile
except NameError:
    def profile(func):
        return func

class DiscreteSOCPFilter:
    def __init__(self, gps, ctrl_mat, input_bound, normalization_vect = np.ones((6,)), slack_weights=[25.0, 250000.0, 25.0], beta_sqrt = [2, 2], state_bound=None, thrust_bound=None, dyn_ext_mat=None):

        self.gps = gps
        self.d_weights = slack_weights # for slack variable, = 2*sqrt(rho) in formulas, 2 components        
        self.beta_sqrt = beta_sqrt # sqrt(beta_i) in formulas

        # get matrices for stability constraint
        self.Ad = ctrl_mat['Ad']
        self.Bd = ctrl_mat['Bd']
        self.Q = ctrl_mat['Q']
        self.R = ctrl_mat['R']
        self.P = ctrl_mat['P']
        self.K = ctrl_mat['K']

        self.norm_z = normalization_vect[:4]      
        self.norm_u = normalization_vect[4:]

        self.input_bound_normalized = input_bound/self.norm_u # normalize the input and state bound for optimization
        self.state_bound = state_bound

        # precompute quantity for stability filter
        W3_mat_comp = self.Ad - self.Bd @ self.K
        self.W3_mat = self.P - W3_mat_comp.T @ self.P @ W3_mat_comp

        # Opt variables and parameters
        self.X = cp.Variable(shape=(7,))
        self.A1 = cp.Parameter(shape=(10, 7))
        self.A2 = cp.Parameter(shape=(9, 7))
        self.A3 = cp.Parameter(shape=(3, 7))
        self.b1 = cp.Parameter(shape=(10,))
        self.b2 = cp.Parameter(shape=(9,))
        self.b3 = np.zeros((3,))
        self.b3[2] = 1
        self.c1 = cp.Parameter(shape=(1, 7))
        self.c2 = cp.Parameter(shape=(1, 7))
        self.c3 = np.zeros((1, 7))
        self.c3[0, 6] = 1
        self.d1 = cp.Parameter()
        self.d2 = cp.Parameter()
        self.d3 = 1
        # put into lists
        As = [self.A1, self.A2, self.A3]
        bs = [self.b1, self.b2, self.b3]
        cs = [self.c1, self.c2, self.c3]
        ds = [self.d1, self.d2, self.d3]

        # state bound
        if state_bound is not None:
            self.h = state_bound['h']
            self.bcon = state_bound['b']
            quantile = state_bound['quantile']
            # precompute values
            h1 = self.h[0:4]
            h2 = self.h[4:8]
            bd1 = np.atleast_2d(self.Bd[0:4, 0]).T
            bd2 = np.atleast_2d(self.Bd[4:8, 1]).T

            self.w_s1 = quantile*np.sqrt(h1.T @ bd1 @ bd1.T @ h1)[0]
            self.w_s2 = quantile*np.sqrt(h2.T @ bd2 @ bd2.T @ h2)[0]

            self.Astate = cp.Parameter(shape=(8, 7))
            self.bstate = cp.Parameter(shape=(8,))
            self.cstate = cp.Parameter(shape=(1, 7))
            self.dstate = cp.Parameter()
            As.append(self.Astate)
            bs.append(self.bstate)
            cs.append(self.cstate)
            ds.append(self.dstate)
        else:
            self.Astate = None
            self.bstate = None
            self.cstate = None
            self.dstate = None

        # create SOC constraints
        m = len(As)
        constraints = [
            cp.SOC(cs[i] @ self.X + ds[i], As[i] @ self.X + bs[i]) for i in range(m)
        ]

        # Add linear constraints: input constraints
        if input_bound is not None: # TODO remove if, input bound always applied due to stability filter
            A_inp = np.zeros((2, 7))
            A_inp[0, 0] = 1.0
            A_inp[1, 1] = 1.0
            constraints = constraints + [A_inp @ self.X <= self.input_bound_normalized]
            constraints = constraints + [-self.input_bound_normalized <= A_inp @ self.X] 

        # bound on thrust, with compensation of dynamic extension
        self.thrust_bound_applied = False
        if thrust_bound is not None:
            self.thrust_bound_applied = True
            # matrices for dynamic extension
            Ad_dyn_ext = dyn_ext_mat['Ad']
            Bd_dyn_ext = dyn_ext_mat['Bd']
            # constraint for dynamic extension
            self.eta = cp.Parameter(shape=(2,))
            # selection_mat = np.zeros((1, 2))
            # selection_mat[0, 0] = 1.0
            unnormalize_mat = np.zeros((2,7))
            unnormalize_mat[0, 0] = self.norm_u[0]
            unnormalize_mat[1, 1] = self.norm_u[1]
            slacking_vect = np.zeros((1, 7))
            slacking_vect[0, 4] = 1.0
            constraints = constraints + [(Ad_dyn_ext @ self.eta + Bd_dyn_ext @ unnormalize_mat @ self.X)[0] <= thrust_bound + slacking_vect @ self.X] # better as SOC constraint??

            
        # define cost function
        self.cost = cp.Parameter(shape=(1, 7))  

        # setup optimization problem      
        self.prob = cp.Problem(cp.Minimize(self.cost @ self.X), constraints)

    @profile
    def compute_feedback_input(self, z, z_ref, v_des, eta= np.zeros((2,)),  x_init=np.zeros((7,)), **kwargs):
        """ Compute u so it can be used in feedback loop
        Args:
        z: flat state to linearize with, from FMPC
        z_ref: reference flat state for stability constraint
        v_des: flat input from FMPC
        x_init=None: initial value for solver"""
    
        # Compute state dependent values
        # remove position and velocity, normalize
        rows_to_remove = [0, 1, 4, 5]
        z_query = np.delete(z, rows_to_remove)
        z_query = z_query/self.norm_z

        gam1 = []
        gam2 = []
        gam3 = []
        gam4 = []
        gam5 = []
        L_gam5 = []
        Linv_gam5 = []
        gp_time = []
        cholesky_time = []
        for i in [0, 1]: #range(len(gp_models)):
            gp_start = time()
            gamma1, gamma2, gamma3, gamma4, gamma5 = get_gammas(z_query, self.gps[i])
            gp_time.append(time() - gp_start)

            chol_start = time()
            L_chol = np.linalg.cholesky(gamma5)
            L_chol_inv = np.linalg.inv(L_chol)
            cholesky_time.append(time() - chol_start)

            gam1.append(gamma1)
            gam2.append(gamma2)
            gam3.append(gamma3)
            gam4.append(gamma4)
            gam5.append(gamma5)
            L_gam5.append(L_chol)
            Linv_gam5.append(L_chol_inv)

        gp_time_total = gp_time[0] + gp_time[1]
        cholesky_time_total = cholesky_time[0] + cholesky_time[1]

        # Compute cost coefficients
        cost_start = time()
        cost = compute_cost(gam1, gam2, gam4, v_des)
        cost_time = time() - cost_start

        # Compute dummy var mats (feedback linearization part)
        dummy_start = time()
        A1, b1, c1, d1 = dummy_var_matrices(gam2, L_gam5, self.d_weights)
        dummy_time = time() - dummy_start

        # Compute stablity filter coeffs
        stab_start = time()
        e_k = z - z_ref
        v_nom = v_des # from equivalence of FMPC with closed form solution
        A2, b2, c2, d2 , A3 = stab_filter_matrices(gam1, gam2, gam3, gam4, L_gam5, Linv_gam5,
                                              self.Q, self.R, self.P, self.K, self.Bd, self.Ad, self.W3_mat, e_k,
                                              self.input_bound_normalized, v_nom, self.beta_sqrt)
        stab_time = time() - stab_start

        # Parameter assignment
        param_start = time()
        self.cost.value = cost
        self.A1.value = A1
        self.b1.value = b1.squeeze()
        self.c1.value = c1
        self.d1.value = d1
        self.A2.value = A2
        self.b2.value = b2.squeeze()
        self.c2.value = c2
        self.d2.value = d2

        self.A3.value = A3 # note: in thesis: A2 and A3 flipped

        # dynamic extension constraint: set previous value of extension states
        if self.thrust_bound_applied:
            self.eta.value = eta

        # # Compute state constraints.
        state_time = 0.0
        if self.state_bound is not None:
            state_start = time()
            Astate, bstate, cstate, dstate = state_con_matrices(z, gam1, gam2, gam3, gam4, L_gam5, Linv_gam5,
                                                                self.h, self.bcon, self.Ad, self.Bd, self.w_s1, self.w_s2)

            self.Astate.value = Astate
            self.bstate.value = bstate.squeeze()
            self.cstate.value = cstate
            self.dstate.value = dstate.squeeze()
            state_time = time() - state_start

        param_time = time() - param_start

        # debugging: print out everything!!
        # print('-----------------------------------------------------------')
        # print('Debugging all variables')
        # print('Inputs z and v')
        # print(z)
        # print(v_des)
        # print('GP 0: Gammas')
        # print(gam1[0])
        # print(gam2[0])
        # print(gam3[0])
        # print(gam4[0])
        # print(gam5[0])
        # print('GP 1: Gammas')
        # print(gam1[1])
        # print(gam2[1])
        # print(gam3[1])
        # print(gam4[1])
        # print(gam5[1])
        # print('Dummy var matrices: A, b, c, d')
        # print(A1)
        # print(b1)
        # print(c1)
        # print(d1)
        # print('Stability constraint matrices: A, b, c, d')
        # print(A2)
        # print(b2)
        # print(c2)
        # print(d2)
        # print('Cost value')
        # print(cost)
        # print('Cholesky decompositions and inverse')
        # print(L_gam5[0])
        # print(Linv_gam5[0])
        # print(L_gam5[1])
        # print(Linv_gam5[1])
        success = False
        logging_dict = {}
        self.X.value = x_init
        mosek_params = {
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-6,  # default 1e-8
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-6,    # primal feasibility, default 1e-8
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-6,    # dual feasibility, default 1e-8
            #'MSK_IPAR_INTPNT_BASIS': 0,  # disable crossover to basic solution
            #'MSK_IPAR_OPTIMIZER': 'MSK_OPTIMIZER_INTPNT',  # ensure interior-point method
            #'MSK_IPAR_NUM_THREADS': 4,  # sometimes single-thread is faster for small problems
        }
        self.prob.solve(solver='MOSEK', warm_start=True, verbose=True, mosek_params=mosek_params)
        #solver_opts = {'iterative_refinement_enable': False,
        #               'tol_feas': 1e-6,
        #               'tol_gap_abs': 1e-6,  # Add this
        #               'tol_gap_rel': 1e-6,  # Add this
        #    }
        #self.prob.solve(solver=cp.CLARABEL, warm_start=True, verbose=True, **solver_opts)

        if 'optimal' in self.prob.status:
            success = True
            # debugging: compute the covariance at this input u: just sample from GP
            # u_opt = self.X.value[0:2]
            # mean0 = gam1[0] + gam2[0].T@u_opt
            # cov0 = gam3[0] + gam4[0].T@u_opt + u_opt.T@gam5[0]@u_opt
            # mean1 = gam1[1] + gam2[1].T@u_opt
            # cov1 = gam3[1] + gam4[1].T@u_opt + u_opt.T@gam5[1]@u_opt
            # means = [mean0, mean1]
            # covs = [cov0, cov1]
            # cost_val = self.cost.value@self.X.value
            # cost_val_lin_part = self.cost.value[0, 0]*self.X.value[0] + self.cost.value[0, 1]*self.X.value[1]
            solve_time = self.prob.solver_stats.solve_time
            # logging_dict['means'] = means
            # logging_dict['covs'] = covs
            # logging_dict['cost'] = cost_val
            # logging_dict['cost_lin'] = cost_val_lin_part
            # logging_dict['q_dummy_val'] = self.X.value[2] # take from X.value directly in fmpc_socp
            # logging_dict['d1_slack'] = self.X.value[3]
            # logging_dict['d2_slack'] = self.X.value[4]
            logging_dict['solve_time'] = solve_time
            logging_dict['gp_time'] = gp_time
            logging_dict['cholesky_time'] = cholesky_time
            logging_dict['cost_time'] = cost_time
            logging_dict['dummy_time'] = dummy_time
            logging_dict['stab_time'] = stab_time
            logging_dict['state_time'] = state_time
            logging_dict['param_time'] = param_time

            return self.X.value[0:2]*self.norm_u, success, self.X.value, logging_dict   
        
        else:
            print('SOCP: Solver failed to find an optimial solution')
            return np.array((0, 0)), 0, 0, 0, 0, 0, [0,0], [0,0]

def get_gammas(z_query, gp_model): 
    query_np = np.hstack((z_query, np.zeros(2))) # zeros as dummy inputs u, to make proper length. get removed in compute_gammas()
    query = torch.from_numpy(query_np).double().unsqueeze(0)
    gamma1, gamma2, gamma3, gamma4, gamma5 = gp_model.model.compute_gammas(query)
    gamma1 = gamma1.numpy().squeeze()
    gamma2 = gamma2.numpy().squeeze()
    gamma3 = gamma3.numpy().squeeze()
    gamma4 = gamma4.numpy().squeeze()
    gamma5 = gamma5.numpy().squeeze()
    return gamma1, gamma2, gamma3, gamma4, gamma5

def compute_cost(gam1, gam2, gam4, v_des):
    gam1_mat = np.vstack((gam1[0], gam1[1]))
    gam2_mat = np.vstack((gam2[0], gam2[1])) # .T or not makes no difference
    cost = 2 * (gam1_mat - v_des.reshape((2,1))).T @ gam2_mat + gam4[0].reshape((1,2)) + gam4[1].reshape((1, 2))
    cost = np.append(cost, np.array([[1.0, 0, 0, 0, 0]]), axis=1)
    return cost

def dummy_var_matrices(gam2, L_gam5, d_weights): # for feedback linearization
    A = np.zeros((10,7))
    A[0, :2] = 2*gam2[0]
    A[1, :2] = 2*gam2[1]
    A[2:4, :2] = 2*L_gam5[0].T
    A[4:6, :2] = 2*L_gam5[1].T
    A[6, 2] = -1.0
    A[7, 3] = d_weights[0]
    A[8, 4] = d_weights[1]
    A[9, 5] = d_weights[2]

    b = np.zeros((10,1))
    b[6, 0] = 1.0

    c = np.zeros((1, 7))
    c[0, 2] = 1.0

    d = 1

    return A, b, c, d

def stab_filter_matrices(gam1,
                         gam2,
                         gam3,
                         gam4,
                         L_gam5,
                         L_gam5_inv,
                         Q, R, P, K, Bd, Ad, W3_mat,
                         e_k,
                         u_max, v_nom, beta_sqrt):
    
    w1 = (2 * e_k.T @ (Ad -Bd@K).T @ P @ Bd)
    # w1_abs = np.abs(w1)
    W2 = (Bd.T @ P @ Bd)
    W2_inv = np.linalg.inv(W2)
    # w3 =  e_k.T @ (Q + K.T @ R @ K) @ e_k - (1e-10) # with DARE reformulation, not used anymore
    # W3_mat is predomputed: P - (A - BK).T P (A-BK)
    w3 = e_k.T @ W3_mat @ e_k - (1e-10) # 1e-10 is the epsilon in the formula
    w4 = 0.5 * W2_inv @ w1
    
    u_t1 = u_max.copy()
    u_t1[0] *= -1.0
    u_t2 = u_max.copy()
    u_t2[1] *= -1.0
    u_test = [u_max, -u_max, u_t1 , u_t2]
    L1 = 2*W2[0,0]*max([(np.abs(gam1[0]-v_nom[0]+w4[0]+ gam2[0].T @ u)+2*np.sqrt(gam3[0]+gam4[0]@u + u@L_gam5[0]@L_gam5[0].T@u)) for u in u_test]) # TODO write as gam5, not L@L.T
    L2 = 2*W2[1,1]*max([(np.abs(gam1[1]-v_nom[1]+w4[1]+ gam2[1].T @ u)+2*np.sqrt(gam3[1]+gam4[1]@u + u@L_gam5[1]@L_gam5[1].T@u)) for u in u_test])

    term_Linv_gam4_0 = 0.5*L_gam5_inv[0] @ gam4[0]
    term_Linv_gam4_1 = 0.5*L_gam5_inv[1] @ gam4[1]

    L1_beta_sqrt = L1* beta_sqrt[0]
    L2_beta_sqrt = L2* beta_sqrt[1]

    A = np.zeros((9,7))    
    A[0:2, 0:2] = L1_beta_sqrt*L_gam5[0]        
    A[4:6, 0:2] = L2_beta_sqrt*L_gam5[1]
    A[8, 6] = 1.0 # dummy variable to extend

    b = np.zeros((9,1))
    b[0:2, 0] = L1_beta_sqrt*term_Linv_gam4_0
    b[4:6, 0] = L2_beta_sqrt*term_Linv_gam4_1

    b[2, 0] = L1_beta_sqrt*np.sqrt(max((0.5*gam3[0] - (term_Linv_gam4_0[0])**2), 1e-10))
    b[3, 0] = L1_beta_sqrt*np.sqrt(max((0.5*gam3[0] - (term_Linv_gam4_0[1])**2), 1e-10))
    b[6, 0] = L2_beta_sqrt*np.sqrt(max((0.5*gam3[1] - (term_Linv_gam4_1[0])**2), 1e-10)) 
    b[7, 0] = L2_beta_sqrt*np.sqrt(max((0.5*gam3[1] - (term_Linv_gam4_1[1])**2), 1e-10))

    c = np.zeros((1, 7))
    c[0, 0:2] = - W2[0,0]*(2*gam1[0] + 2*(w4[0]-v_nom[0]))*gam2[0].T - W2[1,1]*(2*gam1[1] + 2*(w4[1]-v_nom[1]))*gam2[1].T 
    c[0, 3] = 1 # slack variable

    d = w3 + 0.25*w1.T @ W2_inv @ w1 - W2[0,0]*((gam1[0]+w4[0]-v_nom[0])**2) - W2[1,1]*((gam1[1]+w4[1]-v_nom[1])**2) 

    A_dummy = stab_filter_dummy_matrices(gam2, [np.sqrt(W2[0,0]), np.sqrt(W2[1,1])])

    return A, b, c, d, A_dummy

def stab_filter_dummy_matrices(gam2, w2_sqrt):
    A = np.zeros((3, 7))
    A[0, 0:2] = w2_sqrt[0] * gam2[0]
    A[1, 0:2] = w2_sqrt[1] * gam2[1]
    A[2, 6] = -1
    return A

def state_con_matrices(z, gam1, gam2, gam3, gam4, L_gam5, Linv_gam5,
                       h, b_con, Ad, Bd, w_s1, w_s2):
    
    term_Linv_gam4_0 = 0.5*Linv_gam5[0] @ gam4[0]
    term_Linv_gam4_1 = 0.5*Linv_gam5[1] @ gam4[1]
    
    A = np.zeros((8,7))
    A[0:2, 0:2] = w_s1*L_gam5[0].T    
    A[4:6, 0:2] = w_s2*L_gam5[1].T

    b = np.zeros((8,1))
    b[0:2, 0] = w_s1*term_Linv_gam4_0
    b[4:6, 0] = w_s2*term_Linv_gam4_1

    b[2, 0] = w_s1*np.sqrt(max((0.4*gam3[0] - (term_Linv_gam4_0[0])**2), 1e-10)) # distribute gamma3 unevenly for numerical stability
    b[3, 0] = w_s1*np.sqrt(max((0.6*gam3[0] - (term_Linv_gam4_0[1])**2), 1e-10)) # max() is fine, as it only makes constraint more conservative
    b[6, 0] = w_s2*np.sqrt(max((0.4*gam3[1] - (term_Linv_gam4_1[0])**2), 1e-10)) 
    b[7, 0] = w_s2*np.sqrt(max((0.6*gam3[1] - (term_Linv_gam4_1[1])**2), 1e-10))

    c = np.zeros((1, 7))
    c[0, 0:2] = - h.T @ Bd @ np.vstack((gam2[0], gam2[1])) # .T or not makes no difference, see cost function
    c[0, 5] = 1 # slacking it

    d = b_con - h.T @ Ad @ z - h.T @ Bd @ np.vstack((gam1[0], gam1[1]))
    return A, b, c, d
