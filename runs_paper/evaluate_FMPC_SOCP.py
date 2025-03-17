import numpy as np
import pickle
import matplotlib.pyplot as plt

def plot_data(states, time, title, label_x):
    '''plot states 
    input: states/inputs (array: [n_times, n_states])
            time (array: [n_times])'''
    nz = np.shape(states)[1]
    fig, axs = plt.subplots(nz)
    for k in range(nz):
        axs[k].plot(time,states[:, k], color='b', label=' ')
                
    axs[0].set_title(title)
    axs[-1].set(xlabel=label_x)

def plot_data_comparison(states, states_ref, time, title, label_x):
    '''plot states 
    input: states/inputs (array: [n_times, n_states])
            time (array: [n_times])'''
    nz = np.shape(states)[1]
    fig, axs = plt.subplots(nz)
    for k in range(nz):
        axs[k].plot(time,states[:, k], color='b', label='actual')
        axs[k].plot(time,states_ref[:, k], color='r', label='reference')        
    axs[0].set_title(title)
    axs[-1].set(xlabel=label_x)
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')



def evaluateFMPC_SOCP(show_plots = False):

    with open('./temp-data/fmpc_socp_data_quadrotor_traj_tracking.pkl', 'rb') as file:
    # with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/fmpc_data_quadrotor_stabilization.pkl', 'rb') as file:
        data_dict_fmpc = pickle.load(file)
        metrics_dict = data_dict_fmpc['metrics']
        data_dict_fmpc = data_dict_fmpc['trajs_data']
        

    data_dict_fmpc = data_dict_fmpc['controller_data'][0]

    # load data from FMPC
    # u_analytic_noExt = data_dict_fmpc['u_oldFMPC'][0]
    u_analytic_ext = data_dict_fmpc['u_extFT'][0]
    u_socp = data_dict_fmpc['u_extSOCP'][0]
    u = data_dict_fmpc['u'][0]
    gp_means = data_dict_fmpc['gp_means'][0]
    gp_covs = data_dict_fmpc['gp_covs'][0]

    v_des = data_dict_fmpc['v_des'][0]

    d_slack = data_dict_fmpc['socp_slack'][0]
    d_slack2 = data_dict_fmpc['socp_slack2'][0]
    d_slack3 = data_dict_fmpc['socp_slack3'][0]
    q_dummy = data_dict_fmpc['socp_dummy'][0]
    cost_val = data_dict_fmpc['socp_cost'][0]
    cost_val_lin_part = data_dict_fmpc['socp_cost_linPart'][0]
    socp_solve_time = data_dict_fmpc['socp_solve_time'][0]
    thrust_dot = data_dict_fmpc['thrust_dot'][0]

    print('Maximum thrust Tc')
    print(np.max(u, axis=0)[0])
    print('Maximum angle in deg')
    print(np.max(u, axis=0)[1]*180/np.pi)





    if show_plots:
        fig, ax = plt.subplots(8, 2)
        ax[0, 0].plot(range(np.shape(u_analytic_ext)[0]), u_analytic_ext[:, 0], label='analytic, dynamic extension')
        ax[0, 0].plot(range(np.shape(u_analytic_ext)[0]), u_socp[:, 0], label='socp')
        ax[0, 0].set_title('Input extended system: Tc_ddot')
        ax[0, 0].legend()

        ax[1, 0].plot(range(np.shape(u_analytic_ext)[0]), (u_analytic_ext[:, 0] - u_socp[:, 0]), label='difference analytic - socp')
        ax[1, 0].set_title('Tc_ddot: Difference analytic - socp')
  
        ax[0, 1].plot(range(np.shape(u_analytic_ext)[0]), u_analytic_ext[:, 1]*180/np.pi, label='analytic, dynamic extension')
        ax[0, 1].plot(range(np.shape(u_analytic_ext)[0]), u_socp[:, 1]*180/np.pi, label='socp')
        ax[0, 1].set_title('Attitude angle theta')
        ax[0, 1].set_ylabel('degrees')
        ax[0, 1].legend()

        ax[1, 1].plot(range(np.shape(u_analytic_ext)[0]), (u_analytic_ext[:, 1] - u_socp[:, 1])*180/np.pi, label='difference analytic - socp')
        ax[1, 1].set_title('Attitude angle theta: Difference analytic - socp')
        ax[1, 1].set_ylabel('degrees')

        ax[2, 0].plot(range(np.shape(u_analytic_ext)[0]), gp_means[:, 0], label='mean0')
        ax[2, 0].plot(range(np.shape(u_analytic_ext)[0]), v_des[:, 0], label='v0 desired')
        ax[2, 0].set_title('GP0 predictions: mean ')
        ax[2, 0].legend()

        ax[2, 1].plot(range(np.shape(u_analytic_ext)[0]), gp_means[:, 1], label='mean1')
        ax[2, 1].plot(range(np.shape(u_analytic_ext)[0]), v_des[:, 1], label='v1 desired')
        ax[2, 1].set_title('GP1 predictions: mean ')
        ax[2, 1].legend()

        ax[3, 0].plot(range(np.shape(u_analytic_ext)[0]), 2* np.sqrt(gp_covs[:, 0]), label='2stddev0')
        ax[3, 0].set_title('GP0 predictions: 2x standard deviation ')

        ax[3, 1].plot(range(np.shape(u_analytic_ext)[0]), 2* np.sqrt(gp_covs[:, 1]), label='2stddev1')
        ax[3, 1].set_title('GP1 predictions: 2x standard deviation ')

        ax[4, 0].plot(range(np.shape(u_analytic_ext)[0]), d_slack, label='SOCP Slack')
        ax[4, 0].set_title('SOCP slack variable stability')

        ax[4, 1].plot(range(np.shape(u_analytic_ext)[0]), q_dummy, label='SOCP Dummy')
        ax[4, 1].set_title('SOCP dummy variable of FB lin')

        ax[5, 0].plot(range(np.shape(u_analytic_ext)[0]), cost_val, label='SOCP Cost Total')
        ax[5, 0].plot(range(np.shape(u_analytic_ext)[0]), cost_val_lin_part, label='SOCP Cost Linear Term')
        ax[5, 0].plot(range(np.shape(u_analytic_ext)[0]), q_dummy, label='SOCP Cost Quadratic Term')
        ax[5, 0].plot(range(np.shape(u_analytic_ext)[0]), cost_val_lin_part + q_dummy, label='SOCP Cost Total from comp') # sanity check if it all adds up right
        ax[5, 0].set_title('SOCP cost')
        ax[5, 0].legend()

        ax[5, 1].plot(range(np.shape(u_analytic_ext)[0]), socp_solve_time, label='SOCP')
        ax[5, 1].plot((0, np.shape(u_analytic_ext)[0]), (np.mean(socp_solve_time), np.mean(socp_solve_time)), label='SOCP mean')
        ax[5, 1].set_title('Solve times in s')
        ax[5, 1].legend()

        ax[6, 0].plot(range(np.shape(u_analytic_ext)[0]), d_slack2, label='SOCP Slack dynExt')
        ax[6, 0].set_title('SOCP slack variable dynamic extension')

        ax[6, 1].plot(range(np.shape(u_analytic_ext)[0]), d_slack3, label='SOCP Slack state const')
        ax[6, 1].set_title('SOCP slack variable state constraint')

        ax[7, 0].plot(range(np.shape(u_analytic_ext)[0]), thrust_dot, label='Tc_dot')
        ax[7, 0].set_title('Thrust dot in extension')

        plt.show()




if __name__=="__main__":
    evaluateFMPC_SOCP(show_plots=True)