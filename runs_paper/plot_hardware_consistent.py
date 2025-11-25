#!/usr/bin/env python3
"""Plot hardware experiment data using the exact functions from monte_carlo_plotting.py.

This script loads hardware experiment data, formats it to match Monte Carlo data structure,
and uses the EXACT same plotting functions from monte_carlo_plotting.py to ensure
identical styling.

For hardware data, we have single runs (not distributions), but the plotting functions
will handle this correctly by plotting the single trajectory.
"""

import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safe_control_gym.utils.utils import DataVarIndex, Status, load_data

# Import the exact plotting functions from monte_carlo_plotting
from monte_carlo_plotting import (
    plot_tracking_error_distribution,
    plot_position_distribution,
    plot_input_distribution
)


def prepare_hardware_data_for_plotting(file_path, status=None):
    """Load and prepare hardware data in a format compatible with monte_carlo_plotting functions.

    Args:
        file_path (str): Path to CSV file with hardware data
        status (Status): Optional status filter

    Returns:
        dict: Data dictionary formatted like Monte Carlo results (but with single trial)
    """
    # Read the data from the csv file
    data = load_data(file_path)

    if status is not None:
        # Only plot the data that matches the status
        data = data[data[:, DataVarIndex.STATUS] == status.value]

    # Subtract the start time from the time values
    start_time = data[0, DataVarIndex.TIME]
    data[:, DataVarIndex.TIME] -= start_time

    # Extract relevant data
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

    # Create state observations array (n_timesteps, n_states)
    # State: [x, x_dot, z, z_dot, theta, theta_dot]
    # We don't have velocities in the data, so we'll use zeros
    obs = np.column_stack([
        pos_x,
        np.zeros_like(pos_x),  # x_dot (not available)
        pos_z,
        np.zeros_like(pos_z),  # z_dot (not available)
        theta,
        np.zeros_like(theta)   # theta_dot (not available)
    ])

    # Create actions array (n_timesteps, 2)
    action = np.column_stack([thrust_cmd, theta_cmd])

    # Create info dicts with MSE
    info = []
    for i, err in enumerate(pos_error):
        info.append({'mse': err**2})  # Square the error to get MSE

    # Create controller_data with goal states (reference trajectory)
    # Shape: (1 episode, n_timesteps, n_states, 1)
    # We only have reference positions, not full state references
    goal_states = np.zeros((1, len(time), 6, 1))
    goal_states[0, :, 0, 0] = pos_x_des  # Reference x
    goal_states[0, :, 2, 0] = pos_z_des  # Reference z

    controller_data = [{
        'goal_states': goal_states
    }]

    # Create inference time data
    inference_time_data = [inference_time.tolist()]

    # Format as Monte Carlo-style trajectory data (single episode)
    trajs_data = {
        'obs': [obs],  # List of episodes, each is (n_timesteps, n_states)
        'action': [action],  # List of episodes, each is (n_timesteps, n_actions)
        'info': [info],  # List of episodes, each is list of info dicts
        'controller_data': controller_data,
        'inference_time_data': inference_time_data
    }

    return trajs_data



if __name__ == "__main__":
    # Configuration
    status = Status.TRACK_TRAJ  # Only plot tracking portion

    # Specify the data by setting either the run_name or the file_name
    PLOT_CONSTRAINED = False 


    SHADE_STATE_CONSTRAINT = True
    SHADE_INPUT_CONSTRAINT = True
    if PLOT_CONSTRAINED:
        SHOW_FMPC = False
        SHOW_NMPC = False
    else:
        SHOW_FMPC = True 
        SHOW_NMPC = True

    constraint_state = -0.8
    constraint_input = 0.41
    ctrl_freq = 50  # Hz

    # Data paths
    nmpc_path = './hardware_data/unconstrained/data_20250421_211431.csv'
    fmpc_path = './hardware_data/unconstrained/data_20250421_211350.csv'

    if PLOT_CONSTRAINED:
        socp_path = './hardware_data/constrained/data_20250422_143652.csv'
        save_path = './hardware_data/constrained/plots/'
    else:
        socp_path = './hardware_data/unconstrained/data_20250421_211307.csv'
        save_path = './hardware_data/unconstrained/plots/'
    # Prepare data
    results_dict = {}

    if SHOW_NMPC:
        print("Loading NMPC data...")
        results_dict['nmpc'] = {
            'trajs_data': prepare_hardware_data_for_plotting(nmpc_path, status=status)
        }

    if SHOW_FMPC:
        print("Loading FMPC data...")
        results_dict['fmpc'] = {
            'trajs_data': prepare_hardware_data_for_plotting(fmpc_path, status=status)
        }

    print("Loading FMPC+SOCP data...")
    results_dict['fmpc_socp'] = {
        'trajs_data': prepare_hardware_data_for_plotting(socp_path, status=status)
    }

    # Generate plots using the EXACT functions from monte_carlo_plotting.py
    print("\n" + "="*80)
    print("Generating hardware plots using monte_carlo_plotting.py functions")
    print("="*80)

    # Use the exact same functions as Monte Carlo plots
    # They will handle single trajectories correctly (no std bands will be shown)

    plot_position_distribution(results_dict, save_path,
                              is_constrained=SHADE_STATE_CONSTRAINT,
                              constraint_state=constraint_state)

    plot_tracking_error_distribution(results_dict, save_path, ctrl_freq=ctrl_freq)

    plot_input_distribution(results_dict, save_path, ctrl_freq=ctrl_freq,
                           is_constrained=SHADE_INPUT_CONSTRAINT,
                           constraint_input=constraint_input)

    print("\n" + "="*80)
    print("All plots generated successfully!")
    print(f"Saved to: {save_path}")
    print("="*80)
