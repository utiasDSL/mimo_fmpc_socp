import numpy as np
import pickle

import matplotlib.pyplot as plt

"""
Get training data for GP of FMPC_SOCP 
synthetically by evaluating the function on a grid
"""
def plot_data(states, time, title, label_x):
    '''plot states 
    input: states/inputs (array: [n_times, n_states])
            time (array: [n_times])'''
    nz = np.shape(states)[1]
    fig, axs = plt.subplots(nz)
    for k in range(nz):
        axs[k].plot(time,states[:, k], '.', color='b', label=' ')
                
    axs[0].set_title(title)
    axs[-1].set(xlabel=label_x)

# inverse transform of what the GP needs to learn, to double check the other transform
def _get_u_from_flat_states_2D_att_ext(z, v, dyn_pars, g):
    # for system with dynamic extension: u + [Tc_ddot, theta_c]
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

def _get_v_from_Z_and_u_2D_att_ext(z, u, dyn_pars, g):
    # for system with dynamic extension: u + [Tc_ddot, theta_c]
    beta_1 = dyn_pars['beta_1']
    beta_2 = dyn_pars['beta_2']
    alpha_1 =  dyn_pars['alpha_1']
    alpha_2 =  dyn_pars['alpha_2']
    alpha_3 =  dyn_pars['alpha_3']

    term_acc_sqrd = (z[2])**2 + (z[6]+g)**2 # x_ddot^2 + (z_ddot+g)^2
    theta = np.arctan2(z[2], (z[6]+g))
    theta_dot = (z[3]*(z[6]+g)- z[2]*z[7])/term_acc_sqrd
    theta_ddot = alpha_1*theta + alpha_2*theta_dot + alpha_3 * u[1]

    tc = -(beta_2/beta_1) + np.sqrt(term_acc_sqrd)/beta_1
    tc_dot = (1/beta_1)*((1/np.sqrt(term_acc_sqrd))*(z[2]*z[3] + (z[6]+g)*z[7]))

    #precompute sin and cos
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    v_0 = -beta_2*sin_theta*(theta_dot**2) + beta_2*cos_theta*theta_ddot - beta_1*sin_theta*(theta_dot**2)*tc + beta_1*cos_theta*theta_ddot*tc + beta_1*cos_theta*theta_dot*tc_dot + beta_1*cos_theta*theta_dot*tc_dot + beta_1*sin_theta*u[0]
    v_1 = -beta_2*cos_theta*(theta_dot**2) - beta_2*sin_theta*theta_ddot - beta_1*cos_theta*(theta_dot**2)*tc - beta_1*sin_theta*theta_ddot*tc - beta_1*sin_theta*theta_dot*tc_dot - beta_1*sin_theta*theta_dot*tc_dot + beta_1*cos_theta*u[0]    
    return np.array([v_0, v_1])




###################################################################################
################# Main part #######################################################
g=9.8

# 2D Quadrotor Attitude model. TODO: Take from env!
inertial_prop = {}
inertial_prop['alpha_1'] = -140.8
inertial_prop['alpha_2'] = -13.4
inertial_prop['alpha_3'] = 124.8
inertial_prop['beta_1'] = 18.11
inertial_prop['beta_2'] = 3.68


training_data_file = './fgp/gp_train_data_fig8.pkl'
with open(training_data_file, 'rb') as file:
    train_data = pickle.load(file)
inputs_train_list = train_data['inputs']

# dump first few training datasets
# inputs_train_list = inputs_train_list[2:11]
input_data = np.vstack(inputs_train_list)

input_data = input_data[::5, :] # downsample from 50Hz to less

rows_to_remove = [0, 1, 4, 5]
input_data = np.delete(input_data, rows_to_remove, axis=1)

# clip points that are too large, from transient
indices = np.arange(0, np.shape(input_data)[0])
plot_data(input_data, indices, 'Input data raw', 'index')
max_vals = np.array([5, 10, 7, 25, 6, 0.6]) 
min_vals = - max_vals# np.array([-5, -10, -7, -25, -6, -0.6]) 

mask = np.logical_and(input_data >= min_vals, input_data <= max_vals).all(axis=1)

input_data = input_data[mask]
indices = np.arange(0, np.shape(input_data)[0])
# plot_data(input_data, indices, 'Input data clipped', 'index')

normalization_vals = np.max(np.abs(input_data), axis=0)
input_data = input_data/normalization_vals

indices = np.arange(0, np.shape(input_data)[0])
plot_data(input_data, indices, 'Input data', 'index')

# add random offsets around each point:
num_samples = 5
offset_magnitudes = [0.1]

point_noisy_list = []

np.random.seed(9)
for point in input_data:
    for offset_mag in offset_magnitudes:
        for _ in range(num_samples):
            offset_dir = np.random.uniform(-1, 1, 6)
            sample = point + offset_mag*offset_dir
            point_noisy_list.append(sample)

input_data_noisy = np.array(point_noisy_list)

# denormalize back to regular range
input_data_small = input_data_noisy*normalization_vals

# turn it into full z vector again
inputs = np.zeros((np.shape(input_data_small)[0], 10))
inputs[:, 2:4] = input_data_small[:, :2]
inputs[:, 6:10] = input_data_small[:, 2:6]
plot_data(inputs, np.arange(0, np.shape(inputs)[0]), 'Noisy input data: full state vector', 'index')

# get GP targets v from that: 
targets = np.zeros((np.shape(input_data_small)[0], 2))
for i, input in enumerate(inputs): 
    z = input[0:-2]
    u = input[-2:]
    v = _get_v_from_Z_and_u_2D_att_ext(z, u, inertial_prop, g)
    targets[i, :] = v

plot_data(targets, np.arange(0, np.shape(inputs)[0]), 'Target data', 'index')
plt.show()

train_data_dict = {'inputs': inputs, 'targets': targets}

with open('./fgp/gp_train_data_noisyFig8.pkl', 'wb') as file:
    pickle.dump(train_data_dict, file)