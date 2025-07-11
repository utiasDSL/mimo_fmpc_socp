import numpy as np
from numpy.linalg import norm
import cvxpy as cp
import torch
from time import time

# for testing on its own
import os
import pickle
import gpytorch
from safe_control_gym.controllers.mpc.flat_gp_utils import ZeroMeanAffineGP, GaussianProcess
import matplotlib.pyplot as plt

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
        self.A3 = cp.Parameter(shape=(2, 7))
        self.b1 = cp.Parameter(shape=(10,))
        self.b2 = cp.Parameter(shape=(9,))
        self.b3 = np.zeros((2, 1))
        self.c1 = cp.Parameter(shape=(1, 7))
        self.c2 = cp.Parameter(shape=(1, 7))
        self.c3 = cp.Parameter(shape=(1, 7))
        self.d1 = cp.Parameter()
        self.d2 = cp.Parameter()
        self.d3 = 0
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
        for i in [0, 1]: #range(len(gp_models)):
            start_time = time()
            gamma1, gamma2, gamma3, gamma4, gamma5 = get_gammas(z_query, self.gps[i])
            gp_time.append(time()-start_time)
            L_chol = np.linalg.cholesky(gamma5)
            L_chol_inv = np.linalg.inv(L_chol)
            gam1.append(gamma1)
            gam2.append(gamma2)
            gam3.append(gamma3)
            gam4.append(gamma4)
            gam5.append(gamma5)
            L_gam5.append(L_chol)
            Linv_gam5.append(L_chol_inv)
        
        gp_time_total = gp_time[0] + gp_time[1]

        # Compute cost coefficients
        cost = compute_cost(gam1, gam2, gam4, v_des)
        self.cost.value = cost

        # Compute dummy var mats (feedback linearization part)
        A1, b1, c1, d1 = dummy_var_matrices(gam2, L_gam5, self.d_weights)
        self.A1.value = A1
        self.b1.value = b1.squeeze()
        self.c1.value = c1
        self.d1.value = d1

        # Compute stablity filter coeffs
        e_k = z - z_ref
        v_nom = v_des # from equivalence of FMPC with closed form solution
        A2, b2, c2, d2 , A3, c3 = stab_filter_matrices(gam1, gam2, gam3, gam4, L_gam5, Linv_gam5,
                                              self.Q, self.R, self.P, self.K, self.Bd, self.Ad, self.W3_mat, e_k,
                                              self.input_bound_normalized, v_nom, self.beta_sqrt)
        self.A2.value = A2
        self.b2.value = b2.squeeze()
        self.c2.value = c2
        self.d2.value = d2

        self.A3.value = A3
        self.c3.value = c3

        # dynamic extension constraint: set previous value of extension states
        if self.thrust_bound_applied:
            self.eta.value = eta

        # # Compute state constraints.

        if self.state_bound is not None:

            Astate, bstate, cstate, dstate = state_con_matrices(z, gam1, gam2, gam3, gam4, L_gam5, Linv_gam5,
                                                                self.h, self.bcon, self.Ad, self.Bd, self.w_s1, self.w_s2)

            self.Astate.value = Astate
            self.bstate.value = bstate.squeeze()
            self.cstate.value = cstate
            self.dstate.value = dstate.squeeze()

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
        self.prob.solve(solver='MOSEK', warm_start=True, verbose=False)
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
            # solve_time = self.prob.solver_stats.solve_time
            # logging_dict['means'] = means
            # logging_dict['covs'] = covs
            # logging_dict['cost'] = cost_val
            # logging_dict['cost_lin'] = cost_val_lin_part
            # logging_dict['q_dummy_val'] = self.X.value[2] # take from X.value directly in fmpc_socp
            # logging_dict['d1_slack'] = self.X.value[3]
            # logging_dict['d2_slack'] = self.X.value[4]
            # logging_dict['solve_time'] = solve_time
            logging_dict['gp_time'] = gp_time

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
    A[2:4, :2] = 2*L_gam5[0]
    A[4:6, :2] = 2*L_gam5[1]
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
    b[0:2, 0] = -L1_beta_sqrt*term_Linv_gam4_0
    b[4:6, 0] = -L2_beta_sqrt*term_Linv_gam4_1

    b[2, 0] = L1_beta_sqrt*np.sqrt(max((0.5*gam3[0] - (term_Linv_gam4_0[0])**2), 1e-10))
    b[3, 0] = L1_beta_sqrt*np.sqrt(max((0.5*gam3[0] - (term_Linv_gam4_0[1])**2), 1e-10))
    b[6, 0] = L2_beta_sqrt*np.sqrt(max((0.5*gam3[1] - (term_Linv_gam4_1[0])**2), 1e-10)) 
    b[7, 0] = L2_beta_sqrt*np.sqrt(max((0.5*gam3[1] - (term_Linv_gam4_1[1])**2), 1e-10))

    c = np.zeros((1, 7))
    c[0, 0:2] = - W2[0,0]*(2*gam1[0] + 2*(w4[0]-v_nom[0]))*gam2[0].T - W2[1,1]*(2*gam1[1] + 2*(w4[1]-v_nom[1]))*gam2[1].T 
    c[0, 3] = 1 # slack variable

    d = w3 + 0.25*w1.T @ W2_inv @ w1 - W2[0,0]*((gam1[0]+w4[0]-v_nom[0])**2) - W2[1,1]*((gam1[1]+w4[1]-v_nom[1])**2) 

    # tmp = 0.25*w1.T @ W2_inv @ w1
    # tmp2 = W2[0,0]*((gam1[0]+w4[0]-v_nom[0])**2)
    # tmp3 = W2[1,1]*((gam1[1]+w4[1]-v_nom[1])**2)

    A_dummy, c_dummy = stab_filter_dummy_matrices(gam2, [np.sqrt(W2[0,0]), np.sqrt(W2[1,1])])

    # # bound quadratic term
    # bound = np.zeros(2)
    # bound[0] = np.max((np.abs(gam1[0] + gam2[0].T@u_max - v_nom[0]), np.abs(gam1[0] + gam2[0].T@(-u_max) - v_nom[0]))) # here u_min = -u_max, symmetric constraints on u
    # bound[1] = np.max((np.abs(gam1[1] + gam2[1].T@u_max - v_nom[1]), np.abs(gam1[1] + gam2[1].T@(-u_max) - v_nom[1])))
    # w2 = bound.T @ Bd.T @ P @ Bd @ bound

    # term_Linv_gam4_0 = 0.5*L_gam5_inv[0] @ gam4[0]
    # term_Linv_gam4_1 = 0.5*L_gam5_inv[1] @ gam4[1]
    # # if True : #(0.5*gam3[1] - (term_Linv_gam4_1[1])**2) < 0:
    # #     print('Terms Linv*gam4*0.5')
    # #     print(L_gam5[1]@L_gam5[1].T)
    # #     print(L_gam5[1])
    # #     print(L_gam5_inv[1])
    # #     print(gam4[1])
    # #     # print(term_Linv_gam4_0)
    # #     print(term_Linv_gam4_1)
    # #     # print(gam3[0])
    # #     print(gam3[1])
    # #     print((0.5*gam3[1] - (term_Linv_gam4_1[1])**2))
    # #     # exit()

    # # print(Bd)
    # # print(P)
    # # print(Bd.T@P@Bd)

    # # tmp1 = gam1[0] + gam2[0].T@u_max
    # # tmp2 = gam1[0] + gam2[0].T@(-u_max)

    # # tmp3 = gam1[1] + gam2[1].T@u_max
    # # tmp4 = gam1[1] + gam2[1].T@(-u_max)

    # w1_beta_0 = w1_abs[0]*beta_sqrt[0]
    # w1_beta_1 = w1_abs[1]*beta_sqrt[1]

    # A = np.zeros((8,7))
    # A[0:2, 0:2] = w1_beta_0*L_gam5[0]    
    # A[4:6, 0:2] = w1_beta_1*L_gam5[1]

    # b = np.zeros((8,1))
    # b[0:2, 0] = -w1_beta_0*term_Linv_gam4_0
    # b[4:6, 0] = -w1_beta_1*term_Linv_gam4_1

    # b[2, 0] = w1_beta_0*np.sqrt(max((0.4*gam3[0] - (term_Linv_gam4_0[0])**2), 1e-10))
    # b[3, 0] = w1_beta_0*np.sqrt(max((0.6*gam3[0] - (term_Linv_gam4_0[1])**2), 1e-10))
    # b[6, 0] = w1_beta_1*np.sqrt(max((0.4*gam3[1] - (term_Linv_gam4_1[0])**2), 1e-10)) # distribute gamma3 unevenly for numerical stability
    # b[7, 0] = w1_beta_1*np.sqrt(max((0.6*gam3[1] - (term_Linv_gam4_1[1])**2), 1e-10))

    # c = np.zeros((1, 7))
    # c[0, 0:2] = -w1[0]*gam2[0].T -w1[1]*gam2[1].T
    # c[0, 3] = 1

    # d = w3 - w2 -w1[0]*(gam1[0]-v_nom[0]) - w1[1]*(gam1[1]-v_nom[1])  

     

    return A, b, c, d, A_dummy, c_dummy

def stab_filter_dummy_matrices(gam2, w2_sqrt):
    A = np.zeros((2, 7))
    A[0, 0:2] = w2_sqrt[0] * gam2[0]
    A[1, 0:2] = w2_sqrt[1] * gam2[1]

    c = np.zeros((1, 7))
    c[0, 6] = 1
    return A, c

def state_con_matrices(z, gam1, gam2, gam3, gam4, L_gam5, Linv_gam5,
                       h, b_con, Ad, Bd, w_s1, w_s2):
    
    term_Linv_gam4_0 = 0.5*Linv_gam5[0] @ gam4[0]
    term_Linv_gam4_1 = 0.5*Linv_gam5[1] @ gam4[1]
    
    A = np.zeros((8,7))
    A[0:2, 0:2] = w_s1*L_gam5[0]    
    A[4:6, 0:2] = w_s2*L_gam5[1]

    b = np.zeros((8,1))
    b[0:2, 0] = -w_s1*term_Linv_gam4_0
    b[4:6, 0] = -w_s2*term_Linv_gam4_1

    b[2, 0] = w_s1*np.sqrt(max((0.4*gam3[0] - (term_Linv_gam4_0[0])**2), 1e-10)) # distribute gamma3 unevenly for numerical stability
    b[3, 0] = w_s1*np.sqrt(max((0.6*gam3[0] - (term_Linv_gam4_0[1])**2), 1e-10)) # max() is fine, as it only makes constraint more conservative
    b[6, 0] = w_s2*np.sqrt(max((0.4*gam3[1] - (term_Linv_gam4_1[0])**2), 1e-10)) 
    b[7, 0] = w_s2*np.sqrt(max((0.6*gam3[1] - (term_Linv_gam4_1[1])**2), 1e-10))

    c = np.zeros((1, 7))
    c[0, 0:2] = - h.T @ Bd @ np.vstack((gam2[0], gam2[1])) # .T or not makes no difference, see cost function
    c[0, 5] = 1 # slacking it

    d = b_con - h.T @ Ad @ z - h.T @ Bd @ np.vstack((gam1[0], gam1[1]))
    return A, b, c, d

# # for debugging: the transformation that is supposed to be learned with the GP written out analytically
# def _get_u_from_flat_states_2D_att_ext(z, v, dyn_pars, g):
#     # for system with dynamic extension: u + [Tc_ddot, theta_c]
#     beta_1 = dyn_pars['beta_1']
#     beta_2 = dyn_pars['beta_2']
#     alpha_1 =  dyn_pars['alpha_1']
#     alpha_2 =  dyn_pars['alpha_2']
#     alpha_3 =  dyn_pars['alpha_3']

#     term_acc_sqrd = (z[2])**2 + (z[6]+g)**2 # x_ddot^2 + (z_ddot+g)^2
#     theta = np.arctan2(z[2], (z[6]+g))
#     theta_dot = (z[3]*(z[6]+g)- z[2]*z[7])/term_acc_sqrd
#     theta_ddot = 1/term_acc_sqrd * (v[0]*(z[6]+g) - z[2]*v[1]) + (1/(term_acc_sqrd**2)) * (2*(z[6]+g)*z[7] + 2*z[2]*z[3]) * (z[2]*z[7] - z[3]*(z[6]+g))

#     #t = -(beta_2/beta_1) + np.sqrt(term_acc_sqrd)/beta_1
#     p = (1/alpha_3) * (theta_ddot - alpha_1*theta -alpha_2*theta_dot)

#     t_ddot = 1/beta_1 * 1/np.sqrt(term_acc_sqrd)*((z[3]**2 + z[7]**2 + z[2]*v[0] + (z[6]+g)*v[1]) - ((z[2]*z[3] + (z[6]+g)*z[7])**2)/term_acc_sqrd)
#     return np.array([t_ddot, p])

# if __name__ == "__main__":
#     # for analytic reference in debugging
#     g=9.8

#     # 2D Quadrotor Attitude model. TODO: Take from env!
#     inertial_prop = {}
#     inertial_prop['alpha_1'] = -140.8
#     inertial_prop['alpha_2'] = -13.4
#     inertial_prop['alpha_3'] = 124.8
#     inertial_prop['beta_1'] = 18.11
#     inertial_prop['beta_2'] = 3.68

#     # load two GPs
#     output_dir_0 = f'/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/fgp/gp_v0'
#     output_dir_1 = f'/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/fgp/gp_v1'
#     # Check if the folder exists
#     assert os.path.exists(output_dir_0), 'cannot find directory of GP 0'
#     assert os.path.exists(output_dir_1), 'cannot find directory of GP 1'

#     gp_type = ZeroMeanAffineGP
#     likelihood_0 = gpytorch.likelihoods.GaussianLikelihood()
#     gp_0 = GaussianProcess(gp_type, likelihood_0, 1, output_dir_0)
#     gp_0.init_with_hyperparam(output_dir_0)

#     likelihood_1 = gpytorch.likelihoods.GaussianLikelihood()
#     gp_1 = GaussianProcess(gp_type, likelihood_1, 1, output_dir_1)
#     gp_1.init_with_hyperparam(output_dir_1)

#     gps = [gp_0, gp_1]

#     # initialize SOCP Filter
#     filter = DiscreteSOCPFilter('test',gps=gps, input_bound=None) # input_bound=np.array((0.6, 0.3)))

#     # get test points - from evaluation dataset, so that it is a point that makes sense
#     eval_data_file = './examples/mpc/fgp/gp_test_data.pkl' 
#     # eval_data_file = './examples/mpc/fgp/gp_train_data_fig8.pkl'
#     with open(eval_data_file, 'rb') as file:
#         eval_data = pickle.load(file)
#     inputs_eval = eval_data['inputs']
#     targets_eval = eval_data['targets'] 
#     inputs_eval = np.vstack(inputs_eval)
#     targets_eval = np.vstack(targets_eval)

#     z_data = np.transpose(inputs_eval[:, :-2]) #transpose to match data that comes out of FMPC horizon
#     u_data = np.transpose(inputs_eval[:, -2:])
#     v_data = np.transpose(targets_eval[:])

#     # rng = np.random.default_rng(seed=9)
#     # n_datapoints_random =300
#     # z_data = rng.random(( 8, n_datapoints_random))*5
#     # u_data = np.zeros(( 2, n_datapoints_random))
#     # v_data = rng.random(( 2, n_datapoints_random))*10

#     start_idx = 0 #15 #175
#     stop_idx = -1 #50 #215
#     z_data = z_data[:, start_idx:stop_idx]
#     u_data = u_data[:, start_idx:stop_idx]
#     v_data = v_data[:, start_idx:stop_idx]

#     u_socp = np.zeros(np.shape(u_data))
#     covs_run = np.zeros(np.shape(u_data))
#     means_run = np.zeros(np.shape(u_data))
#     success_list = []
#     d_sf_list = []
#     q_dummy_list = []

#     for point_idx in range(np.shape(z_data)[1]):
#         z_test = z_data[:,point_idx]
#         v_test = v_data[:,point_idx]    
#         # compute forward
#         u, success, d_sf, q_dummy, cost_val, cost_val_lin_part,  means, covs = filter.compute_feedback_input(z_test, z_test, v_test)
#         u_socp[:, point_idx] = u
#         success_list.append(success)
#         d_sf_list.append(d_sf)
#         q_dummy_list.append(q_dummy)
#         covs_run[:, point_idx] = covs
#         means_run[:, point_idx] = means
    
#     u_analytic = np.zeros(np.shape(u_data))
#     for point_idx in range(np.shape(z_data)[1]):
#         z_test = z_data[:,point_idx]
#         v_test = v_data[:,point_idx]    
#         u = _get_u_from_flat_states_2D_att_ext(z_test, v_test, inertial_prop, g)
#         u_analytic[:, point_idx] = u
    
#     # plot test data
#     fig, ax = plt.subplots(2, 3) 
#     t = np.arange(0, np.shape(u_data)[1])
#     # First subplot
#     ax[0, 0].plot(t, u_data[0, :], label='Test input u0' ) 
#     ax[0, 0].plot(t, u_socp[0, :], label='SOCP result u0' ) 
#     ax[0, 0].plot(t, u_analytic[0, :], label='analytic reference u0' ) 
#     ax[0, 0].set_title("First component u0")
#     ax[0, 0].set_xlabel("datapoint")
#     ax[0, 0].set_ylabel("Tc_ddot")
#     ax[0, 0].legend()

#     ax[1, 0].plot(t, u_data[1, :], label='Test input u1' ) 
#     ax[1, 0].plot(t, u_socp[1, :], label='SOCP result u1' )
#     ax[1, 0].plot(t, u_analytic[1, :], label='analytic reference u1' ) 
#     ax[1, 0].set_title("Second Component u1")
#     ax[1, 0].set_xlabel("datapoint")
#     ax[1, 0].set_ylabel("Theta_c")
#     ax[1, 0].legend()

#     ax[0, 1].plot(t, means_run[0, :], label='Mean GP0' )
#     ax[0, 1].plot(t, v_data[0, :], label='v_0' )   
#     ax[0, 1].set_title("Mean GP0")
#     ax[0, 1].set_xlabel("datapoint")
#     ax[0, 1].set_ylabel("Mean")
#     ax[0, 1].legend()

#     ax[1, 1].plot(t, means_run[1, :], label='Mean GP1' ) 
#     ax[1, 1].plot(t, v_data[1, :], label='v_1' )  
#     ax[1, 1].set_title("Mean GP1")
#     ax[1, 1].set_xlabel("datapoint")
#     ax[1, 1].set_ylabel("Mean")
#     ax[1, 1].legend()

#     ax[0, 2].plot(t, 2*np.sqrt(covs_run[0, :]), label='2x standard deviation GP0' )  
#     ax[0, 2].set_title("2x Standard Deviation GP0")
#     ax[0, 2].set_xlabel("datapoint")
#     ax[0, 2].set_ylabel("2x Std Dev")
#     ax[0, 2].legend()

#     ax[1, 2].plot(t, 2*np.sqrt(covs_run[1, :]), label='2x standard deviation GP1' )  
#     ax[1, 2].set_title("2x Standard Deviation GP1")
#     ax[1, 2].set_xlabel("datapoint")
#     ax[1, 2].set_ylabel("2x Std Dev")
#     ax[1, 2].legend()

#     plt.show()