# runs experiments for paper
# plots the data for paper
import os
import argparse
from runs_paper.mpc_experiment_paper import run
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42



parser = argparse.ArgumentParser(description="Run the script in different modes.")
parser.add_argument(
    '--mode',
    type=str,
    default='normal',  # Default unconstrained lemniscate
    choices=['normal', 'constrained'],
    help='Choose the mode: normal or constrained (default is normal).'
)

args = parser.parse_args()

if args.mode == 'normal':
    print("Running unconstrained lemniscate.")
    yaml_file_base = './config_overrides_fast/quadrotor_2D_attitude_tracking.yaml'
    yaml_file_nmpc = './config_overrides_fast/mpc_quadrotor_2D_attitude_tracking.yaml'
    yaml_file_fmpc = './config_overrides_fast/fmpc_ext_quadrotor_2D_attitude_tracking.yaml'
    yaml_file_fmpc_socp = './config_overrides_fast/fmpc_socp_quadrotor_2D_attitude_tracking.yaml'
    SHADE_STATE_CONSTRAINT = False
    SHADE_INPUT_CONSTRAINT = False
    RUN_FMPC=True
    SHOW_FMPC = True
    MAKE_THRUST_CLOSEUP = False

elif args.mode == 'constrained':
    print("Running constrained lemniscate.")
    yaml_file_base = './config_overrides_constrained/quadrotor_2D_attitude_tracking.yaml'
    yaml_file_nmpc = './config_overrides_constrained/mpc_quadrotor_2D_attitude_tracking.yaml'
    yaml_file_fmpc = './config_overrides_constrained/fmpc_ext_quadrotor_2D_attitude_tracking.yaml'
    yaml_file_fmpc_socp = './config_overrides_constrained/fmpc_socp_quadrotor_2D_attitude_tracking.yaml'
    SHADE_STATE_CONSTRAINT = True
    SHADE_INPUT_CONSTRAINT = True
    RUN_FMPC=False
    SHOW_FMPC = False
    MAKE_THRUST_CLOSEUP = True



######### Parameters ###############################
RUN_NMPC=True
# RUN_FMPC=True
RUN_FMPC_SOCP=True

GUI = False

ctrl_freq = 50
sample_time = 1/ctrl_freq
num_loops = 2


constraint_state = -0.8
constraint_input = 0.435

fig_width = 5.8 # inches
fig_height = 5.8/1.4
plt.rcParams.update({
    'font.size': 10, 
    "text.usetex": True,            # Use LaTeX for text rendering
    "font.family": "serif",         # Match LaTeX font (e.g., Computer Modern)
    "legend.fontsize": 8,           # Legend size
    })
alpha_lines = 0.6
alpha_constraint = 0.2

#########################################
data_path_nmpc = './temp-data/mpc_data_quadrotor_traj_tracking.pkl'
data_path_fmpc = './temp-data/fmpc_ext_data_quadrotor_traj_tracking.pkl'
data_path_fmpc_socp = './temp-data/fmpc_socp_data_quadrotor_traj_tracking.pkl'

if RUN_NMPC:
    if os.path.exists(data_path_nmpc):
        os.remove(data_path_nmpc)
        print(f"{data_path_nmpc} deleted successfully.")
    else:
        print(f"{data_path_nmpc} does not exist.")
    
    # run controller
    run(gui=GUI, save_data=True, algo = 'mpc', yaml_base = yaml_file_base, yaml_ctrl = yaml_file_nmpc)

if RUN_FMPC:
    if os.path.exists(data_path_fmpc):
        os.remove(data_path_fmpc)
        print(f"{data_path_fmpc} deleted successfully.")
    else:
        print(f"{data_path_fmpc} does not exist.")
    
    # run controller
    run(gui=GUI, save_data=True, algo = 'fmpc_ext', yaml_base = yaml_file_base, yaml_ctrl = yaml_file_fmpc)

if RUN_FMPC_SOCP:
    if os.path.exists(data_path_fmpc_socp):
        os.remove(data_path_fmpc_socp)
        print(f"{data_path_fmpc_socp} deleted successfully.")
    else:
        print(f"{data_path_fmpc_socp} does not exist.")
    
    # run controller
    run(gui=GUI, save_data=True, algo = 'fmpc_socp', yaml_base = yaml_file_base, yaml_ctrl = yaml_file_fmpc_socp)

#######################################
# extract data
def extract_data(data_file):
    with open(data_file, 'rb') as file:
        data_dict = pickle.load(file)
    metrics = data_dict['metrics']
    traj_data = data_dict['trajs_data']
    states = traj_data['obs'][0]
    # state_mpc = traj_data_mpc['state'][0] # exactly the same as 'obs' for our case (no noise I guess)

    mse_dict = []
    for info in traj_data['info'][0]:
        if 'mse' in info:
            mse_dict.append(info.get('mse'))
        else:
            mse_dict.append(0) # only occurs at initial timestep as far as I can tell
    error = np.array(mse_dict)
    rmse = metrics['rmse']
    # rmse_from_error = np.sqrt(np.mean(error_mpc))
    # rmse_diff = rmse_mpc - rmse_from_error # sanity check: order of e-5 NOTE: rounding errors?

    inference_time = np.array(traj_data['inference_time_data'])
    # diff_time = np.mean(inference_time_mpc) - metrics_mpc['avarage_inference_time'] # = 0, so fine

    state_ref = np.array(traj_data['controller_data'][0]['goal_states'])

    # inputs
    action = traj_data['action'][0]

    ctrl_data = traj_data['controller_data'][0]
    if 'u_ext' in ctrl_data:
        action_ext = np.array(ctrl_data['u_ext'][0]) # FMPC
        gp_time=0
    elif 'u_extSOCP' in ctrl_data:
        action_ext = np.array(ctrl_data['u_extSOCP'][0]) # FMPC_SOCP
        gp_time = np.array(ctrl_data['gp_time'][0])
    else:
        action_ext = 0
        gp_time = 0
    return states, error, inference_time, rmse, state_ref, action, action_ext, gp_time
if RUN_NMPC:
    state_mpc, error_mpc, inf_time_mpc, rmse_mpc, state_ref_mpc, action_mpc, _, _ = extract_data(data_path_nmpc)
if RUN_FMPC:
    state_x_fmpc, error_fmpc, inf_time_fmpc, rmse_fmpc, state_ref_fmpc, action_fmpc, action_ext_fmpc, _ = extract_data(data_path_fmpc) 
if RUN_FMPC_SOCP:
    state_x_fmpc_socp, error_fmpc_socp, inf_time_fmpc_socp, rmse_fmpc_socp, state_ref_fmpc_socp, action_fmpc_socp, action_ext_fmpc_socp, gp_time_socp = extract_data(data_path_fmpc_socp)   



###########################################################
# data visualization
# Define TUM colors
tum_blue = '#0065BD'
tum_blue_1 = '#98C6EA' # lightest
tum_blue_2 = '#64A0C8'
tum_blue_3 = '#0073CF'
tum_blue_4 = '#005293'
tum_blue_5 = '#003359' # darkest
# accent colors
tum_green = '#A2AD00'
tum_orange = '#E37222'
tum_ivory = '#DAD7CB'
# diagram colors
tum_dia_violet = '#69085A'
tum_dia_dark_blue = '#0F1B5F'
tum_dia_turquoise = '#00778A'
tum_dia_dark_green = '#007C30'
tum_dia_light_green = '#679A1D'
tum_dia_light_yellow = '#FFDC00'
tum_dia_dark_yellow = '#F9BA00'
tum_dia_dark_orange = '#D64C13'
tum_dia_red = '#C4071B'
tum_dia_dark_red = '#9C0D16'

# define color and labels
ref_color = 'black'
mpc_color = tum_blue_3
fmpc_color = tum_dia_dark_green
fmpc_socp_color = tum_dia_dark_orange

ref_label = '_nolegend_' #'reference'
mpc_label = 'NMPC (true dynamics)'
fmpc_label = 'FMPC (true dynamics)'
fmpc_socp_label = 'FMPC+GP+SOCP (ours)'

linewidth = 2.5

limits_x = [-1.1, 1.1]
limits_y = [0.4, 1.6]

# plot of figure 8 in 2D space
plt.figure(figsize=(fig_width, fig_height))
plt.plot(state_ref_mpc[0, :301, 0, 0], state_ref_mpc[0, :301, 2, 0], linestyle = 'dashed', color=ref_color, label=ref_label, linewidth=linewidth, alpha=alpha_lines)
plt.plot(state_mpc[:, 0], state_mpc[:, 2], color=mpc_color, label=mpc_label, linewidth=linewidth, alpha=alpha_lines)
if SHOW_FMPC:   
    plt.plot(state_x_fmpc[:, 0], state_x_fmpc[:, 2], color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
plt.plot(state_x_fmpc_socp[:, 0], state_x_fmpc_socp[:, 2], color=fmpc_socp_color, label=fmpc_socp_label, linewidth=linewidth, alpha=alpha_lines)
plt.legend()
plt.xlabel(r'Position x (m)')
plt.ylabel(r'Position z (m)')
plt.xlim(limits_x)
plt.ylim(limits_y)
plt.grid()
if SHADE_STATE_CONSTRAINT:
    plt.axvspan(limits_x[0], constraint_state, color=tum_dia_red, alpha=alpha_constraint)
plt.savefig("./plots/fig8.pdf", format="pdf", bbox_inches=None)

# plot errors over time
time = np.arange(0, np.shape(error_mpc)[0]*sample_time, sample_time )
plt.figure(figsize=(fig_width, fig_height))
plt.plot(time, np.sqrt(error_mpc), color=mpc_color, label=mpc_label, linewidth=linewidth, alpha=alpha_lines)
if SHOW_FMPC:
    plt.plot(time, np.sqrt(error_fmpc), color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
plt.plot(time, np.sqrt(error_fmpc_socp), color=fmpc_socp_color, label=fmpc_socp_label, linewidth=linewidth, alpha=alpha_lines)
plt.legend()
plt.xlabel(r'Time (s)')
plt.ylabel(r'Tracking error (m)')
plt.grid()
plt.savefig("./plots_thesis/tracking_error.pdf", format="pdf", bbox_inches=None)

# generate a bunch of metrics on tracking error
end_idx_first_loop = int(np.shape(state_mpc)[0]/num_loops)
def compute_tracking_error(error, end_idx_first_loop):
    mean_track_err = np.mean(np.sqrt(error))
    loop1_track_err = np.mean(np.sqrt(error[:end_idx_first_loop]))
    loop2_track_err = np.mean(np.sqrt(error[end_idx_first_loop:]))
    return mean_track_err, loop1_track_err, loop2_track_err
if RUN_NMPC:
    mean_track_err_mpc, loop1_track_err_mpc, loop2_track_err_mpc = compute_tracking_error(error_mpc, end_idx_first_loop)
if RUN_FMPC:
    mean_track_err_fmpc, loop1_track_err_fmpc, loop2_track_err_fmpc = compute_tracking_error(error_fmpc, end_idx_first_loop)
else: 
    mean_track_err_fmpc = 999
    loop1_track_err_fmpc = 999
    loop2_track_err_fmpc = 999
if RUN_FMPC_SOCP:
    mean_track_err_fmpc_socp, loop1_track_err_fmpc_socp, loop2_track_err_fmpc_socp = compute_tracking_error(error_fmpc_socp, end_idx_first_loop)

print('\nTracking Error: mean(sqrt(sum of squares at each timestep))')
print('                     NMPC   |  FMPC   | FMPC+SOCP')
print(' average track_err: {:.2f}mm | {:.2f}mm | {:.2f}mm'.format(mean_track_err_mpc*1000, mean_track_err_fmpc*1000, mean_track_err_fmpc_socp*1000))
print('1st loop track_err: {:.2f}mm | {:.2f}mm | {:.2f}mm'.format(loop1_track_err_mpc*1000, loop1_track_err_fmpc*1000, loop1_track_err_fmpc_socp*1000))
print('2nd loop track_err: {:.2f}mm | {:.2f}mm | {:.2f}mm'.format(loop2_track_err_mpc*1000, loop2_track_err_fmpc*1000, loop2_track_err_fmpc_socp*1000))

print('\nRMSE: sqrt(mean(sum of squares at each timestep))')
print('                     NMPC   |  FMPC   | FMPC+SOCP')
# print('      average RMSE: {:.2f}mm | {:.2f}mm | {:.2f}mm'.format(rmse_mpc*1000, rmse_fmpc*1000, rmse_fmpc_socp*1000))

##################################################################################
# Inputs
limits_y1 = [0, 0.5]
time = np.arange(0, np.shape(action_mpc)[0]*sample_time, sample_time )
fig, ax = plt.subplots(2, figsize=(fig_width, fig_height))
ax[0].plot(time, action_mpc[:, 0], color=mpc_color, label=mpc_label, linewidth=linewidth, alpha=alpha_lines)
if SHOW_FMPC:       
    ax[0].plot(time, action_fmpc[:, 0], color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
ax[0].plot(time, action_fmpc_socp[:, 0], color=fmpc_socp_color, label=fmpc_socp_label, linewidth=linewidth, alpha=alpha_lines)
ax[0].set_ylabel(r'Thrust $T_c$ (N)')
# ax[0].set_ylim(limits_y1)
ax[0].grid()
ax[1].plot(time, action_mpc[:, 1], color=mpc_color, label=mpc_label, linewidth=linewidth, alpha=alpha_lines)
if SHOW_FMPC:
    ax[1].plot(time, action_fmpc[:, 1], color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
ax[1].plot(time, action_fmpc_socp[:, 1], color=fmpc_socp_color, label=fmpc_socp_label, linewidth=linewidth, alpha=alpha_lines)
ax[1].set_ylabel(r'Angle $\theta_c$ (rad)')
ax[1].grid()
ax[1].set_xlabel(r'Time (s)')
if SHADE_INPUT_CONSTRAINT:
    ax[0].axhspan(constraint_input, limits_y1[1], color=tum_dia_red, alpha=alpha_constraint)
ax[1].legend(loc="upper right")

plt.savefig("./plots_thesis/inputs.pdf", format="pdf", bbox_inches=None)

# Input closeup for constrained case
if MAKE_THRUST_CLOSEUP:
    limits_y1 = [0.4, 0.45]
    limits_x = [0, 6]
    time = np.arange(0, np.shape(action_mpc)[0]*sample_time, sample_time )
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    ax.plot(time, action_mpc[:, 0], color=mpc_color, label=mpc_label, linewidth=linewidth, alpha=alpha_lines)
    if SHOW_FMPC:       
        ax[0].plot(time, action_fmpc[:, 0], color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
    ax.plot(time, action_fmpc_socp[:, 0], color=fmpc_socp_color, label=fmpc_socp_label, linewidth=linewidth, alpha=alpha_lines)
    ax.set_ylabel(r'Thrust $T_c$ (N)')
    ax.set_ylim(limits_y1)
    ax.set_xlim(limits_x)
    ax.grid()
    if SHADE_INPUT_CONSTRAINT:
        ax.axhspan(constraint_input, limits_y1[1], color=tum_dia_red, alpha=alpha_constraint)
    ax.legend() #(loc="upper right")
    plt.savefig("./plots_thesis/input_closeup.pdf", format="pdf", bbox_inches=None) 

# # Tc_ddot in Flatness based controllers
# time = np.arange(0, np.shape(action_ext_fmpc)[0]*sample_time, sample_time )
# plt.figure(figsize=(fig_width, fig_height))
# # plt.plot(time, np.sqrt(error_mpc), color=mpc_color, label=mpc_label, linewidth=linewidth)
# plt.plot(time, action_ext_fmpc[:, 0], color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
# plt.plot(time, action_ext_fmpc_socp[:, 0], color=fmpc_socp_color, label=fmpc_socp_label, linewidth=linewidth, alpha=alpha_lines)
# plt.legend()
# plt.xlabel('time in s')
# plt.ylabel(r'$\ddot{T_c}$ in $\frac{N}{s^2}$')
# plt.grid()

# print('\nMaximum of inputs')
# print('               NMPC   |  FMPC   | FMPC+SOCP')
# print('Thrust: {:.5f}N   | {:.5f}N   | {:.5f}N'.format(np.max(action_mpc[:, 0]), np.max(action_fmpc[:, 0]), np.max(action_fmpc_socp[:, 0])))
# print(' Angle: {:.2f}rad | {:.2f}rad | {:.2f}rad'.format(np.max(action_mpc[:, 1]), np.max(action_fmpc[:, 1]), np.max(action_fmpc_socp[:, 1])))

# print('\nMaximum of extended input')
# print('               NMPC   |  FMPC   | FMPC+SOCP')
# print('Thrust_ddot: ------  | {:.2f}N/s^2 | {:.2f}N/s^2'.format(np.max(action_ext_fmpc[:, 0]), np.max(action_ext_fmpc_socp[:, 0])))


##################################################################################
# plot inference time over time
time = np.arange(0, np.shape(inf_time_mpc)[1]*sample_time, sample_time )
# plt.figure()
# plt.plot(time, inf_time_mpc.T, color=mpc_color, label=mpc_label, linewidth=linewidth)
# plt.plot(time, inf_time_fmpc.T, color=fmpc_color, label=fmpc_label, linewidth=linewidth)
# plt.plot(time, inf_time_fmpc_socp.T, color=fmpc_socp_color, label=fmpc_socp_label, linewidth=linewidth)
# plt.legend()
# plt.xlabel('time in s')
# plt.ylabel('inference time in s')
# plt.grid()

# print('\nInference Time of each controller')
# print('               NMPC   |  FMPC   | FMPC+SOCP')
# print('average time: {:.2f}ms | {:.2f}ms | {:.2f}ms'.format(np.mean(inf_time_mpc)*1000, np.mean(inf_time_fmpc)*1000, np.mean(inf_time_fmpc_socp)*1000))
# print('maximum time: {:.2f}ms | {:.2f}ms | {:.2f}ms'.format(np.max(inf_time_mpc)*1000, np.max(inf_time_fmpc)*1000, np.max(inf_time_fmpc_socp)*1000))

# print('gp_infe time: {:.2f}ms | {:.2f}ms | {:.2f}ms'.format(0, 0, np.mean(gp_time_socp)*1000))


######################################################################################
# print('Velocities of the quadrotor along the trajectory')

# vel_mpc = np.sqrt(state_mpc[:, 1]**2 + state_mpc[:, 3]**2)
# vel_fmpc = np.sqrt(state_x_fmpc[:, 1]**2 + state_x_fmpc[:, 3]**2)
# vel_fmpc_socp = np.sqrt(state_x_fmpc_socp[:, 1]**2 + state_x_fmpc_socp[:, 3]**2)

# # plot velocity over time
# # time = np.arange(0, np.shape(error_mpc)[0]*sample_time, sample_time )
# # plt.figure()
# # plt.plot(time, vel_mpc, color=mpc_color, label=mpc_label, linewidth=linewidth)
# # plt.plot(time, vel_fmpc, color=fmpc_color, label=fmpc_label, linewidth=linewidth)
# # plt.plot(time, vel_fmpc_socp, color=fmpc_socp_color, label=fmpc_socp_label, linewidth=linewidth)
# # plt.legend()
# # plt.xlabel('time in s')
# # plt.ylabel('velocity in m/s')
# # plt.grid()

# print('\nVelocity on trajectory')
# print('                   NMPC   |  FMPC   | FMPC+SOCP')
# print('average velocity: {:.2f}m/s | {:.2f}m/s | {:.2f}m/s'.format(np.mean(vel_mpc), np.mean(vel_fmpc), np.mean(vel_fmpc_socp)))
# print('maximum velocity: {:.2f}m/s | {:.2f}m/s | {:.2f}m/s'.format(np.max(vel_mpc), np.max(vel_fmpc), np.max(vel_fmpc_socp)))





plt.show()

dummy = 0