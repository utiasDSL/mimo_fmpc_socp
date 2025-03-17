import numpy as np
import pickle

import matplotlib.pyplot as plt


from mpc_quad_gp_training_data import run

import os
import yaml

"""
Get training data for GP of FMPC_SOCP 
from simulations in Safe Control Gym with FMPC_EXT: The FMPC with dynamic extension, as similar as possible to what the GP has to learn
(here exact transformations are known)
"""
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

def run_sim_scg(additional=''):
    run(gui=False, save_data=True, ALGO='fmpc_ext', ADDITIONAL=additional)

def load_data_scg():
    with open('./temp-gp-training-data/fmpc_ext_data_quadrotor_traj_tracking.pkl', 'rb') as file:
        data_dict = pickle.load(file)
        data_dict = data_dict['trajs_data']['controller_data'][0]
        # print(data_dict.keys())
        # exit()      
    z = data_dict['z_inp'][0] 
    v = data_dict['v_inp'][0] 
    u_ext = data_dict['u_ext'][0] 

    return z, u_ext, v

def get_one_dataset(additional, PLOT_RUN):
    run_sim_scg(additional)
    z, u_bar, v = load_data_scg()

    # get data from yaml file
    SYS = 'quadrotor_2D_attitude'
    TASK = 'tracking'
    ADDITIONAL = additional
    assert os.path.exists(f'./config_overrides/{SYS}/{SYS}_{TASK}{ADDITIONAL}.yaml'), f'./config_overrides/{SYS}/{SYS}_{TASK}{ADDITIONAL}.yaml does not exist'
    with open(f'./config_overrides/{SYS}/{SYS}_{TASK}{ADDITIONAL}.yaml') as file:
        data_yaml = yaml.safe_load(file)
    episode_len_sec = data_yaml['task_config']['episode_len_sec']
    ctrl_freq = data_yaml['task_config']['ctrl_freq']
    num_cycles = data_yaml['task_config']['task_info']['num_cycles']

    # find start/stop index of cycle 0.5 - 1.5 (full figure8 but starting in the middle)
    use_cycle_num = 1.5
    start_index = 0 #int((episode_len_sec/num_cycles) * ctrl_freq *(use_cycle_num-1))
    stop_index = int((episode_len_sec/num_cycles) * ctrl_freq *(use_cycle_num))
    # cut down data
    u_bar_data = u_bar[start_index:stop_index, :]
    z_data = z[start_index:stop_index, :]
    v_data = v[start_index:stop_index, :]

    # assemble into GP training data
    x_train = np.hstack((z_data, u_bar_data))

    if PLOT_RUN:
        # plotting
        # times = np.linspace(episode_len_sec/num_cycles*(use_cycle_num-1), episode_len_sec/num_cycles*use_cycle_num, np.shape(u_bar_data)[0])
        times = np.linspace(0, episode_len_sec, np.shape(u_bar_data)[0])
        plot_data(z_data, times, 'Flat States Z', 'time')
        plot_data(v_data, times, 'Flat Input Trajectory V', 'time')
        plot_data(u_bar_data, times, 'GP training data input u_bar', 'time' )
        plt.show()
    return x_train, v_data


###################################################################################
################# Main part #######################################################
PLOT_RUN = False
additional_list_train = ['_tr2', '_tr3', '_tr4', '_tr5', '_tr6', '_tr7', '_tr8', '_tr9', '_tr10', '_tr11']
additional_list_test = ['_te1', '_te2', '_te3', '_te4']


# Training data ###################################################################
inputs = []
targets = []

for additional in additional_list_train:
    x_train, v_data = get_one_dataset(additional, PLOT_RUN)
    inputs.append(x_train)
    targets.append(v_data)

inputs_arr = np.vstack(inputs)
targets_arr = np.vstack(targets)

indices = np.arange(0, np.shape(inputs_arr)[0])
plot_data(inputs_arr, indices, 'GP training inputs z and u', 'index')
plot_data(targets_arr, indices, 'GP training targets v', 'index')
plt.show()

train_data_dict = {'inputs': inputs, 'targets': targets}

with open('./fgp/gp_train_data_fig8.pkl', 'wb') as file:
    pickle.dump(train_data_dict, file)

# Test data #######################################################################

inputs = []
targets = []

for additional in additional_list_test:
    x_train, v_data = get_one_dataset(additional, PLOT_RUN)
    inputs.append(x_train)
    targets.append(v_data)

inputs_arr = np.vstack(inputs)
targets_arr = np.vstack(targets)

indices = np.arange(0, np.shape(inputs_arr)[0])
plot_data(inputs_arr, indices, 'GP test inputs', 'index')
plot_data(targets_arr, indices, 'GP test targets', 'index')
plt.show()


test_data_dict = {'inputs': inputs, 'targets': targets}

with open('./fgp/gp_test_data.pkl', 'wb') as file:
    pickle.dump(test_data_dict, file)



