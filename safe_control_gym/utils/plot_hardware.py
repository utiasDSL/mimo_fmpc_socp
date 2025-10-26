import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import tikzplotlib

from safe_control_gym.utils.utils import DataVarIndex, Status, load_data 



class Plotter:
    """A class that plots the recorded data."""

    def __init__(self, save_fig=False):
        self.save_fig = save_fig
        self.rmse = None 

        # Create a dictionary to match the actual value with the desired value
        self.match_desired = {
            DataVarIndex.POS_X: DataVarIndex.DES_POS_X,
            DataVarIndex.POS_Y: DataVarIndex.DES_POS_Y,
            DataVarIndex.POS_Z: DataVarIndex.DES_POS_Z,
            DataVarIndex.ROLL: DataVarIndex.CMD_ROLL,
            DataVarIndex.PITCH: DataVarIndex.CMD_PITCH,
            DataVarIndex.YAW: DataVarIndex.CMD_YAW,
            DataVarIndex.VEL_X: DataVarIndex.DES_VEL_X,
            DataVarIndex.VEL_Y: DataVarIndex.DES_VEL_Y,
            DataVarIndex.VEL_Z: DataVarIndex.DES_VEL_Z,
            DataVarIndex.PITCH_PRED: DataVarIndex.PITCH_ACT,
        }
        
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y']

    def prepare_data_thesis(self, file_path, status=None):
        # Read the data from the csv file
        data = load_data(file_path)


        if status is not None:
            # Only plot the data that matches the status
            data = data[data[:, DataVarIndex.STATUS] == status.value]

        # Subtract the start time from the time values
        start_time = data[0, DataVarIndex.TIME]
        data[:, DataVarIndex.TIME] -= start_time
        
        pos_x = data[:, DataVarIndex.POS_X]
        pos_z = data[:, DataVarIndex.POS_Z]
        pos_x_des = data[:, DataVarIndex.DES_POS_X]
        pos_z_des = data[:, DataVarIndex.DES_POS_Z]
        theta = data[:, DataVarIndex.PITCH]
        theta_cmd = data[:, DataVarIndex.CMD_PITCH]
        thrust_cmd = data[:, DataVarIndex.CMD_FORCE]
        pos_error = data[:, DataVarIndex.POS_ERROR]
        inference_time = data[:, DataVarIndex.INFERENCE_TIME]
        time = data[:, DataVarIndex.TIME]

        return pos_x, pos_z, pos_x_des, pos_z_des, theta, theta_cmd, thrust_cmd, pos_error, inference_time, time
        

if __name__ == "__main__":
    # Plot the entire trajectory or just the tracking part
    # status = None
    status = Status.TRACK_TRAJ
    # status = Status.HOVER

    # Specify the data by setting either the run_name or the file_name

    SHADE_STATE_CONSTRAINT = True
    SHADE_INPUT_CONSTRAINT = True
    SHOW_FMPC = True 
    SHOW_NMPC = True
    # Plot the data
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    constraint_state = -0.8
    constraint_input = 0.41

    fig_width = 5.8 # inches
    fig_height = 5.8/1.4
    plt.rcParams.update({
        'font.size': 10, 
        "text.usetex": True,            # Use LaTeX for text rendering
        "font.family": "serif",         # Match LaTeX font (e.g., Computer Modern)
        "legend.fontsize": 8,           # Legend size
        # "pgf.texsystem": "pdflatex"
        })
    alpha_lines = 0.6
    alpha_constraint = 0.2
    
    plotter = Plotter(save_fig=True)
    
    nmpc_path = '/home/ahall/Documents/UofT/papers/mimo_fmpc_socp_gp_ecc/mimo_fmpc_socp/runs_paper/hardware_data/Run5_unconstrainedDark2/data_20250421_211431.csv'
    fmpc_path = '/home/ahall/Documents/UofT/papers/mimo_fmpc_socp_gp_ecc/mimo_fmpc_socp/runs_paper/hardware_data/Run5_unconstrainedDark2/data_20250421_211350.csv'
    socp_path = '/home/ahall/Documents/UofT/papers/mimo_fmpc_socp_gp_ecc/mimo_fmpc_socp/runs_paper/hardware_data/Run5_unconstrainedDark2/data_20250421_211307.csv'

    save_path_1 = '/home/ahall/Documents/UofT/papers/mimo_fmpc_socp_gp_ecc/mimo_fmpc_socp/runs_paper/hardware_data/Run5_unconstrainedDark2/plots/'

    pos_x_socp, pos_z_socp, pos_x_des_socp, pos_z_des_socp, theta_socp, theta_cmd_socp, thrust_cmd_socp, pos_error_socp, inference_time_socp, time_socp = plotter.prepare_data_thesis(socp_path, status=status)
    pos_x_fmpc, pos_z_fmpc, pos_x_des_fmpc, pos_z_des_fmpc, theta_fmpc, theta_cmd_fmpc, thrust_cmd_fmpc, pos_error_fmpc, inference_time_fmpc, time_fmpc = plotter.prepare_data_thesis(fmpc_path, status=status)
    pos_x_nmpc, pos_z_nmpc, pos_x_des_nmpc, pos_z_des_nmpc, theta_nmpc, theta_cmd_nmpc, thrust_cmd_nmpc, pos_error_nmpc, inference_time_nmpc, time_nmpc = plotter.prepare_data_thesis(nmpc_path, status=status)




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
    nmpc_color = tum_blue_3
    fmpc_color = tum_dia_dark_green
    fmpc_socp_color = tum_dia_dark_orange

    ref_label = '_nolegend_' #'reference'
    nmpc_label = 'NMPC (fitted model)'
    fmpc_label = 'FMPC (fitted model)'
    fmpc_socp_label = 'FMPC+GP+SOCP'

    linewidth = 2.5

    limits_x = [-1.1, 1.1]
    limits_y = [0.4, 1.6]

    # plot of figure 8 in 2D space
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(pos_x_des_socp, pos_z_des_socp, linestyle = 'dashed', color=ref_color, label=ref_label, linewidth=linewidth, alpha=alpha_lines)
    if SHOW_NMPC:
        plt.plot(pos_x_nmpc, pos_z_nmpc, color=nmpc_color, label=nmpc_label, linewidth=linewidth, alpha=alpha_lines)  
    if SHOW_FMPC:
        plt.plot(pos_x_fmpc, pos_z_fmpc, color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
    plt.plot(pos_x_socp, pos_z_socp, color=fmpc_socp_color, label=fmpc_socp_label, linewidth=linewidth, alpha=alpha_lines)
    plt.legend()
    plt.xlabel(r'Position x (m)')
    plt.ylabel(r'Position z (m)')
    plt.xlim(limits_x)
    # plt.ylim(limits_y)
    plt.grid()
    if SHADE_STATE_CONSTRAINT:
        plt.axvspan(limits_x[0], constraint_state, color=tum_dia_red, alpha=alpha_constraint)
    plt.savefig(save_path_1 + 'fig8.pdf', format="pdf", bbox_inches=None)

    # plot errors over time
    plt.figure(figsize=(fig_width, fig_height))
    if SHOW_NMPC:
        plt.plot(time_nmpc, pos_error_nmpc, color=nmpc_color, label=nmpc_label, linewidth=linewidth, alpha=alpha_lines)
    if SHOW_FMPC:
        plt.plot(time_fmpc, pos_error_fmpc, color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
    plt.plot(time_socp, pos_error_socp, color=fmpc_socp_color, label=fmpc_socp_label, linewidth=linewidth, alpha=alpha_lines)
    plt.legend()
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'Tracking error (m)')
    plt.grid()
    plt.savefig(save_path_1 + 'tracking_error.pdf', format="pdf", bbox_inches=None)

    tikzplotlib.save("test.tex")


    # Inputs
    limits_y1 = [0.15, 0.45]
    fig, ax = plt.subplots(2, figsize=(fig_width, fig_height))
    if SHOW_NMPC:
        ax[0].plot(time_nmpc, thrust_cmd_nmpc, color=nmpc_color, label=nmpc_label, linewidth=linewidth, alpha=alpha_lines)
    if SHOW_FMPC:
        ax[0].plot(time_fmpc, thrust_cmd_fmpc, color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
    ax[0].plot(time_socp, thrust_cmd_socp, color=fmpc_socp_color, label=fmpc_socp_label, linewidth=linewidth, alpha=alpha_lines)
    ax[0].set_ylabel(r'Thrust $T_c$ (N)')
    ax[0].set_ylim(limits_y1)
    ax[0].grid()
    if SHOW_NMPC:
        ax[1].plot(time_nmpc, theta_cmd_nmpc, color=nmpc_color, label=nmpc_label, linewidth=linewidth, alpha=alpha_lines)
    if SHOW_FMPC:
        ax[1].plot(time_fmpc, theta_cmd_fmpc, color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
    ax[1].plot(time_socp, theta_cmd_socp, color=fmpc_socp_color, label=fmpc_socp_label, linewidth=linewidth, alpha=alpha_lines)
    ax[1].set_ylabel(r'Angle $\theta_c$ (rad)')
    ax[1].grid()
    ax[1].set_xlabel(r'Time (s)')
    if SHADE_INPUT_CONSTRAINT:
        ax[0].axhspan(constraint_input, limits_y1[1], color=tum_dia_red, alpha=alpha_constraint)
    ax[1].legend(loc="upper right")

    plt.savefig(save_path_1 + 'inputs.pdf', format="pdf", bbox_inches=None)

    plt.show()

    # inference time statistics
    # means
    inference_time_nmpc_mean = np.mean(inference_time_nmpc)
    inference_time_fmpc_mean = np.mean(inference_time_fmpc)  
    inference_time_socp_mean = np.mean(inference_time_socp)

    # maximum
    inference_time_nmpc_max = np.max(inference_time_nmpc)
    inference_time_fmpc_max = np.max(inference_time_fmpc)
    inference_time_socp_max = np.max(inference_time_socp)

    print("Inference time statistics in milliseconds")
    print("NMPC mean: ", inference_time_nmpc_mean*1000)
    print("FMPC mean: ", inference_time_fmpc_mean*1000)
    print("SOCP mean: ", inference_time_socp_mean*1000)
    print("NMPC max: ", inference_time_nmpc_max*1000)
    print("FMPC max: ", inference_time_fmpc_max*1000)
    print("SOCP max: ", inference_time_socp_max*1000)