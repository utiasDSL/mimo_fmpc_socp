"""Monte Carlo plotting and analysis functions.

This module contains all plotting and statistical analysis functions for Monte Carlo
experiment results. It has no dependencies on safe_control_gym, only numpy and matplotlib.

These functions are used by both:
- run_exp_paper_monte_carlo.py (for generating plots during experiments)
- regenerate_plots.py (for regenerating plots from saved data)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib PDF font type for better compatibility
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({
    'font.size': 4, 
    "text.usetex": True,            # Use LaTeX for text rendering
    "font.family": "serif",         # Match LaTeX font (e.g., Computer Modern)
    "legend.fontsize": 4,           # Legend size
    # "pgf.texsystem": "pdflatex"
    })


def compute_second_half_rmse(trajs_data):
    """Compute RMSE for only the second half of each trajectory.

    Args:
        trajs_data (dict): Trajectory data containing info with mse values

    Returns:
        float: Mean RMSE across all episodes (second half only)
        float: Standard deviation of RMSE across episodes
        list: Per-episode RMSE values (second half only)
    """
    second_half_rmse_list = []

    if trajs_data == {}:
        return 0.0, 0.0, []
    else:

    # Process each episode
        for ep_info in trajs_data['info']:
            # Extract MSE values from info dicts
            mse_values = []
            for info in ep_info:
                if 'mse' in info:
                    mse_values.append(info['mse'])

            if len(mse_values) == 0:
                continue

            # Split at midpoint and take second half
            midpoint = len(mse_values) // 2
            second_half_mse = mse_values[midpoint:]

            # Compute RMSE for second half
            if len(second_half_mse) > 0:
                rmse_second_half = float(np.sqrt(np.mean(second_half_mse)))
                second_half_rmse_list.append(rmse_second_half)

        # Compute statistics across all episodes
        if len(second_half_rmse_list) > 0:
            mean_rmse = np.mean(second_half_rmse_list)
            std_rmse = np.std(second_half_rmse_list)
        else:
            mean_rmse = 0.0
            std_rmse = 0.0

        return mean_rmse, std_rmse, second_half_rmse_list


def extract_timing_statistics(trajs_data, controller_name):
    """Extract detailed timing statistics from trajectory data.

    Args:
        trajs_data (dict): Trajectory data containing controller_data with timing info
        controller_name (str): Name of controller for identifying available timing data

    Returns:
        dict: Dictionary with timing statistics (mean and std for each timing component)
    """
    timing_stats = {}

    # Check if controller_data exists
    if 'controller_data' not in trajs_data or len(trajs_data['controller_data']) == 0:
        return timing_stats

    # Iterate over ALL trials, not just the first one
    # MPC solve time (available for all controllers)
    if 'mpc_solve_time' in trajs_data['controller_data'][0]:
        all_mpc_times = []
        for trial_ctrl_data in trajs_data['controller_data']:
            if 'mpc_solve_time' in trial_ctrl_data:
                for episode_data in trial_ctrl_data['mpc_solve_time']:
                    all_mpc_times.extend(episode_data)
        if len(all_mpc_times) > 0:
            timing_stats['mpc_solve'] = {
                'mean': np.mean(all_mpc_times) * 1000,  # Convert to ms
                'std': np.std(all_mpc_times) * 1000
            }

    # FMPC_SOCP and GPMPC specific timings
    if controller_name.lower() in ['fmpc_socp', 'fmpc+socp', 'gpmpc']:
        # SOCP solve time
        if 'socp_solve_time' in trajs_data['controller_data'][0]:
            all_socp_times = []
            for trial_ctrl_data in trajs_data['controller_data']:
                if 'socp_solve_time' in trial_ctrl_data:
                    for episode_data in trial_ctrl_data['socp_solve_time']:
                        all_socp_times.extend(episode_data)
            if len(all_socp_times) > 0:
                timing_stats['socp_solve'] = {
                    'mean': np.mean(all_socp_times) * 1000,
                    'std': np.std(all_socp_times) * 1000
                }

        # GP inference time (list of 2 values per timestep)
        if 'gp_time' in trajs_data['controller_data'][0]:
            all_gp_times = []
            for trial_ctrl_data in trajs_data['controller_data']:
                if 'gp_time' in trial_ctrl_data:
                    for episode_data in trial_ctrl_data['gp_time']:
                        for timestep_gp_times in episode_data:
                            # Sum the two GP times per timestep
                            if isinstance(timestep_gp_times, (list, np.ndarray)):
                                all_gp_times.append(sum(timestep_gp_times))
                            else:
                                all_gp_times.append(timestep_gp_times)
            if len(all_gp_times) > 0:
                timing_stats['gp_inference'] = {
                    'mean': np.mean(all_gp_times) * 1000,
                    'std': np.std(all_gp_times) * 1000
                }

        # Observer time
        if 'observer_time' in trajs_data['controller_data'][0]:
            all_observer_times = []
            for trial_ctrl_data in trajs_data['controller_data']:
                if 'observer_time' in trial_ctrl_data:
                    for episode_data in trial_ctrl_data['observer_time']:
                        all_observer_times.extend(episode_data)
            if len(all_observer_times) > 0:
                timing_stats['observer'] = {
                    'mean': np.mean(all_observer_times) * 1000,
                    'std': np.std(all_observer_times) * 1000
                }

        # Flat transformation time
        if 'flat_transform_time' in trajs_data['controller_data'][0]:
            all_transform_times = []
            for trial_ctrl_data in trajs_data['controller_data']:
                if 'flat_transform_time' in trial_ctrl_data:
                    for episode_data in trial_ctrl_data['flat_transform_time']:
                        all_transform_times.extend(episode_data)
            if len(all_transform_times) > 0:
                timing_stats['flat_transform'] = {
                    'mean': np.mean(all_transform_times) * 1000,
                    'std': np.std(all_transform_times) * 1000
                }

        # Dynamic extension time
        if 'dyn_ext_time' in trajs_data['controller_data'][0]:
            all_dyn_ext_times = []
            for trial_ctrl_data in trajs_data['controller_data']:
                if 'dyn_ext_time' in trial_ctrl_data:
                    for episode_data in trial_ctrl_data['dyn_ext_time']:
                        all_dyn_ext_times.extend(episode_data)
            if len(all_dyn_ext_times) > 0:
                timing_stats['dyn_extension'] = {
                    'mean': np.mean(all_dyn_ext_times) * 1000,
                    'std': np.std(all_dyn_ext_times) * 1000
                }

    return timing_stats


def print_timing_breakdown_table(results_dict):
    """Print detailed timing breakdown for each controller.

    Args:
        results_dict (dict): Dictionary containing results for each controller
    """
    print('\n' + '='*80)
    print('TIMING BREAKDOWN (ms)')
    print('='*80)

    # Extract timing data for each controller
    nmpc_timing = {}
    fmpc_timing = {}
    fmpc_socp_timing = {}
    gpmpc_timing = {}

    if 'nmpc' in results_dict:
        nmpc_timing = extract_timing_statistics(
            results_dict['nmpc']['trajs_data'], 'nmpc'
        )

    if 'fmpc' in results_dict:
        fmpc_timing = extract_timing_statistics(
            results_dict['fmpc']['trajs_data'], 'fmpc'
        )

    if 'fmpc_socp' in results_dict:
        fmpc_socp_timing = extract_timing_statistics(
            results_dict['fmpc_socp']['trajs_data'], 'fmpc_socp'
        )

    if 'gpmpc' in results_dict:
        gpmpc_timing = extract_timing_statistics(
            results_dict['gpmpc']['trajs_data'], 'gpmpc'
        )

    # Print table
    print(f'\n{"Component":<25} | {"NMPC":>20} | {"FMPC":>20} | {"FMPC+SOCP":>20} | {"GPMPC":>20}')
    print('-'*115)

    # MPC solve time
    nmpc_mpc = nmpc_timing.get('mpc_solve', {'mean': 0, 'std': 0})
    fmpc_mpc = fmpc_timing.get('mpc_solve', {'mean': 0, 'std': 0})
    fmpc_socp_mpc = fmpc_socp_timing.get('mpc_solve', {'mean': 0, 'std': 0})
    gpmpc_mpc = gpmpc_timing.get('mpc_solve', {'mean': 0, 'std': 0})

    print(f'{"MPC Solve":<25} | {nmpc_mpc["mean"]:>8.3f} ± {nmpc_mpc["std"]:>7.3f} | '
          f'{fmpc_mpc["mean"]:>8.3f} ± {fmpc_mpc["std"]:>7.3f} | '
          f'{fmpc_socp_mpc["mean"]:>8.3f} ± {fmpc_socp_mpc["std"]:>7.3f} | '
          f'{gpmpc_mpc["mean"]:>8.3f} ± {gpmpc_mpc["std"]:>7.3f}')

    # SOCP solve time (only for FMPC+SOCP)
    if 'socp_solve' in fmpc_socp_timing:
        socp = fmpc_socp_timing['socp_solve']
        print(f'{"SOCP Solve":<25} | {"—":>20} | {"—":>20} | '
              f'{socp["mean"]:>8.3f} ± {socp["std"]:>7.3f} | '
              f'{"—":>20}')

    # GP inference time (for FMPC+SOCP and GPMPC)
    if 'gp_inference' in fmpc_socp_timing or 'gp_inference' in gpmpc_timing:
        fmpc_socp_gp = fmpc_socp_timing.get('gp_inference', {'mean': 0, 'std': 0})
        gpmpc_gp = gpmpc_timing.get('gp_inference', {'mean': 0, 'std': 0})
        gp_str_socp = f'{fmpc_socp_gp["mean"]:>8.3f} ± {fmpc_socp_gp["std"]:>7.3f}' if 'gp_inference' in fmpc_socp_timing else "—"
        gp_str_gpmpc = f'{gpmpc_gp["mean"]:>8.3f} ± {gpmpc_gp["std"]:>7.3f}' if 'gp_inference' in gpmpc_timing else "—"
        print(f'{"GP Inference":<25} | {"—":>20} | {"—":>20} | '
              f'{gp_str_socp:>20} | '
              f'{gp_str_gpmpc:>20}')

    # Observer time (for FMPC+SOCP and GPMPC)
    if 'observer' in fmpc_socp_timing or 'observer' in gpmpc_timing:
        fmpc_socp_obs = fmpc_socp_timing.get('observer', {'mean': 0, 'std': 0})
        gpmpc_obs = gpmpc_timing.get('observer', {'mean': 0, 'std': 0})
        obs_str_socp = f'{fmpc_socp_obs["mean"]:>8.3f} ± {fmpc_socp_obs["std"]:>7.3f}' if 'observer' in fmpc_socp_timing else "—"
        obs_str_gpmpc = f'{gpmpc_obs["mean"]:>8.3f} ± {gpmpc_obs["std"]:>7.3f}' if 'observer' in gpmpc_timing else "—"
        print(f'{"Flat State Observer":<25} | {"—":>20} | {"—":>20} | '
              f'{obs_str_socp:>20} | '
              f'{obs_str_gpmpc:>20}')

    # Flat transformation time (for FMPC+SOCP and GPMPC)
    if 'flat_transform' in fmpc_socp_timing or 'flat_transform' in gpmpc_timing:
        fmpc_socp_ft = fmpc_socp_timing.get('flat_transform', {'mean': 0, 'std': 0})
        gpmpc_ft = gpmpc_timing.get('flat_transform', {'mean': 0, 'std': 0})
        ft_str_socp = f'{fmpc_socp_ft["mean"]:>8.3f} ± {fmpc_socp_ft["std"]:>7.3f}' if 'flat_transform' in fmpc_socp_timing else "—"
        ft_str_gpmpc = f'{gpmpc_ft["mean"]:>8.3f} ± {gpmpc_ft["std"]:>7.3f}' if 'flat_transform' in gpmpc_timing else "—"
        print(f'{"Flat Transformation":<25} | {"—":>20} | {"—":>20} | '
              f'{ft_str_socp:>20} | '
              f'{ft_str_gpmpc:>20}')

    # Dynamic extension time (for FMPC+SOCP and GPMPC)
    if 'dyn_extension' in fmpc_socp_timing or 'dyn_extension' in gpmpc_timing:
        fmpc_socp_de = fmpc_socp_timing.get('dyn_extension', {'mean': 0, 'std': 0})
        gpmpc_de = gpmpc_timing.get('dyn_extension', {'mean': 0, 'std': 0})
        de_str_socp = f'{fmpc_socp_de["mean"]:>8.3f} ± {fmpc_socp_de["std"]:>7.3f}' if 'dyn_extension' in fmpc_socp_timing else "—"
        de_str_gpmpc = f'{gpmpc_de["mean"]:>8.3f} ± {gpmpc_de["std"]:>7.3f}' if 'dyn_extension' in gpmpc_timing else "—"
        print(f'{"Dynamic Extension":<25} | {"—":>20} | {"—":>20} | '
              f'{de_str_socp:>20} | '
              f'{de_str_gpmpc:>20}')

    print('-'*115)
    print()


def print_summary_table(results_dict):
    """Print a summary table comparing all controllers.

    Args:
        results_dict (dict): Dictionary containing results for each controller
    """
    print('\n' + '='*100)
    print('SUMMARY: Monte Carlo Experiment Results')
    print('='*100)

    # Extract metrics
    nmpc_metrics = results_dict.get('nmpc', {}).get('metrics', {})
    fmpc_metrics = results_dict.get('fmpc', {}).get('metrics', {})
    fmpc_socp_metrics = results_dict.get('fmpc_socp', {}).get('metrics', {})
    gpmpc_metrics = results_dict.get('gpmpc', {}).get('metrics', {})

    # Print trial statistics first
    print('\n' + '='*100)
    print('TRIAL STATISTICS')
    print('='*100)
    print(f'\n{"Statistic":<30} | {"NMPC":>15} | {"FMPC":>15} | {"FMPC+SOCP":>15} | {"GPMPC":>15}')
    print('-'*100)

    # Number of trials
    nmpc_n_trials = nmpc_metrics.get('n_trials', 0)
    fmpc_n_trials = fmpc_metrics.get('n_trials', 0)
    fmpc_socp_n_trials = fmpc_socp_metrics.get('n_trials', 0)
    gpmpc_n_trials = gpmpc_metrics.get('n_trials', 0)

    print(f'{"Total Trials":<30} | {nmpc_n_trials:>15d} | '
          f'{fmpc_n_trials:>15d} | '
          f'{fmpc_socp_n_trials:>15d} | '
          f'{gpmpc_n_trials:>15d}')

    # Successful trials
    nmpc_n_success = nmpc_metrics.get('n_successful', 0)
    fmpc_n_success = fmpc_metrics.get('n_successful', 0)
    fmpc_socp_n_success = fmpc_socp_metrics.get('n_successful', 0)
    gpmpc_n_success = gpmpc_metrics.get('n_successful', 0)

    print(f'{"Successful Trials":<30} | {nmpc_n_success:>15d} | '
          f'{fmpc_n_success:>15d} | '
          f'{fmpc_socp_n_success:>15d} | '
          f'{gpmpc_n_success:>15d}')

    # Failed trials
    nmpc_n_failed = nmpc_metrics.get('n_failed', 0)
    fmpc_n_failed = fmpc_metrics.get('n_failed', 0)
    fmpc_socp_n_failed = fmpc_socp_metrics.get('n_failed', 0)
    gpmpc_n_failed = gpmpc_metrics.get('n_failed', 0)

    print(f'{"Failed Trials":<30} | {nmpc_n_failed:>15d} | '
          f'{fmpc_n_failed:>15d} | '
          f'{fmpc_socp_n_failed:>15d} | '
          f'{gpmpc_n_failed:>15d}')

    # Success rate
    nmpc_success_rate = nmpc_metrics.get('success_rate', 0)
    fmpc_success_rate = fmpc_metrics.get('success_rate', 0)
    fmpc_socp_success_rate = fmpc_socp_metrics.get('success_rate', 0)
    gpmpc_success_rate = gpmpc_metrics.get('success_rate', 0)

    print(f'{"Success Rate (%)":<30} | {nmpc_success_rate*100:>15.1f} | '
          f'{fmpc_success_rate*100:>15.1f} | '
          f'{fmpc_socp_success_rate*100:>15.1f} | '
          f'{gpmpc_success_rate*100:>15.1f}')

    print('-'*100)

    # Performance metrics (computed over successful trials only)
    print('\n' + '='*100)
    print('PERFORMANCE METRICS (Successful Trials Only)')
    print('='*100)
    print(f'\n{"Metric":<30} | {"NMPC":>15} | {"FMPC":>15} | {"FMPC+SOCP":>15} | {"GPMPC":>15}')
    print('-'*100)

    # Full trajectory RMSE
    print(f'{"Average RMSE (m)":<30} | {nmpc_metrics.get("average_rmse", 0):>15.4f} | '
          f'{fmpc_metrics.get("average_rmse", 0):>15.4f} | '
          f'{fmpc_socp_metrics.get("average_rmse", 0):>15.4f} | '
          f'{gpmpc_metrics.get("average_rmse", 0):>15.4f}')

    print(f'{"RMSE Std Dev (m)":<30} | {nmpc_metrics.get("rmse_std", 0):>15.4f} | '
          f'{fmpc_metrics.get("rmse_std", 0):>15.4f} | '
          f'{fmpc_socp_metrics.get("rmse_std", 0):>15.4f} | '
          f'{gpmpc_metrics.get("rmse_std", 0):>15.4f}')

    # Second half RMSE
    nmpc_second_half = (0.0, 0.0)
    fmpc_second_half = (0.0, 0.0)
    fmpc_socp_second_half = (0.0, 0.0)
    gpmpc_second_half = (0.0, 0.0)

    if 'nmpc' in results_dict:
        nmpc_second_half = compute_second_half_rmse(results_dict['nmpc']['trajs_data'])[:2]
    if 'fmpc' in results_dict:
        fmpc_second_half = compute_second_half_rmse(results_dict['fmpc']['trajs_data'])[:2]
    if 'fmpc_socp' in results_dict:
        fmpc_socp_second_half = compute_second_half_rmse(results_dict['fmpc_socp']['trajs_data'])[:2]
    if 'gpmpc' in results_dict:
        gpmpc_second_half = compute_second_half_rmse(results_dict['gpmpc']['trajs_data'])[:2]

    print(f'{"RMSE Second Half (m)":<30} | {nmpc_second_half[0]:>15.4f} | '
          f'{fmpc_second_half[0]:>15.4f} | '
          f'{fmpc_socp_second_half[0]:>15.4f} | '
          f'{gpmpc_second_half[0]:>15.4f}')

    print(f'{"RMSE Second Half Std (m)":<30} | {nmpc_second_half[1]:>15.4f} | '
          f'{fmpc_second_half[1]:>15.4f} | '
          f'{fmpc_socp_second_half[1]:>15.4f} | '
          f'{gpmpc_second_half[1]:>15.4f}')

    # Inference time - compute from ALL timestep measurements, not trial means
    # Extract all individual timestep measurements
    def get_all_inference_times(trajs_data):
        """Extract all individual timestep inference times from trajectory data."""
        all_times = []
        if 'inference_time_data' in trajs_data:
            for trial_times in trajs_data['inference_time_data']:
                all_times.extend(trial_times)
        return all_times if len(all_times) > 0 else [0]

    nmpc_all_times = get_all_inference_times(results_dict.get('nmpc', {}).get('trajs_data', {}))
    fmpc_all_times = get_all_inference_times(results_dict.get('fmpc', {}).get('trajs_data', {}))
    fmpc_socp_all_times = get_all_inference_times(results_dict.get('fmpc_socp', {}).get('trajs_data', {}))
    gpmpc_all_times = get_all_inference_times(results_dict.get('gpmpc', {}).get('trajs_data', {}))

    nmpc_time = np.mean(nmpc_all_times)
    nmpc_time_std = np.std(nmpc_all_times)
    fmpc_time = np.mean(fmpc_all_times)
    fmpc_time_std = np.std(fmpc_all_times)
    fmpc_socp_time = np.mean(fmpc_socp_all_times)
    fmpc_socp_time_std = np.std(fmpc_socp_all_times)
    gpmpc_time = np.mean(gpmpc_all_times)
    gpmpc_time_std = np.std(gpmpc_all_times)

    print(f'{"Avg Inference Time (ms)":<30} | {nmpc_time*1000:>7.2f} ± {nmpc_time_std*1000:>5.2f} | '
          f'{fmpc_time*1000:>7.2f} ± {fmpc_time_std*1000:>5.2f} | '
          f'{fmpc_socp_time*1000:>7.2f} ± {fmpc_socp_time_std*1000:>5.2f} | '
          f'{gpmpc_time*1000:>7.2f} ± {gpmpc_time_std*1000:>5.2f}')

    # Constraint violations
    print(f'{"Failure Rate (%)":<30} | {nmpc_metrics.get("failure_rate", 0)*100:>15.2f} | '
          f'{fmpc_metrics.get("failure_rate", 0)*100:>15.2f} | '
          f'{fmpc_socp_metrics.get("failure_rate", 0)*100:>15.2f} | '
          f'{gpmpc_metrics.get("failure_rate", 0)*100:>15.2f}')

    print(f'{"Avg Constraint Violations":<30} | {nmpc_metrics.get("average_constraint_violation", 0):>15.2f} | '
          f'{fmpc_metrics.get("average_constraint_violation", 0):>15.2f} | '
          f'{fmpc_socp_metrics.get("average_constraint_violation", 0):>15.2f} | '
          f'{gpmpc_metrics.get("average_constraint_violation", 0):>15.2f}')

    print('-'*100)
    print()

    # Print detailed timing breakdown
    print_timing_breakdown_table(results_dict)


def plot_inference_time_violin(results_dict, output_dir='./monte_carlo_results/normal'):
    """Create violin plot comparing inference times across controllers.

    Args:
        results_dict (dict): Dictionary containing results for each controller
        output_dir (str): Directory to save the plot
    """
    # Define TUM colors
    tum_blue_3 = '#0073CF'
    tum_dia_dark_green = '#007C30'
    tum_dia_dark_orange = '#D64C13'

    # Extract inference time data for each controller
    data_to_plot = []
    labels = []
    colors = []

    # NMPC
    if 'nmpc' in results_dict:
        nmpc_trajs = results_dict['nmpc']['trajs_data']
        nmpc_inf_times = []
        for traj_inf_time in nmpc_trajs['inference_time_data']:
            nmpc_inf_times.extend(traj_inf_time)  # Flatten all timesteps from all trials
        if len(nmpc_inf_times) > 0:
            data_to_plot.append(np.array(nmpc_inf_times) * 1000)  # Convert to ms
            labels.append('NMPC')
            colors.append(tum_blue_3)

    # FMPC (optional)
    if 'fmpc' in results_dict:
        fmpc_trajs = results_dict['fmpc']['trajs_data']
        fmpc_inf_times = []
        for traj_inf_time in fmpc_trajs['inference_time_data']:
            fmpc_inf_times.extend(traj_inf_time)
        if len(fmpc_inf_times) > 0:
            data_to_plot.append(np.array(fmpc_inf_times) * 1000)  # Convert to ms
            labels.append('FMPC')
            colors.append(tum_dia_dark_green)

    # FMPC+SOCP
    if 'fmpc_socp' in results_dict:
        fmpc_socp_trajs = results_dict['fmpc_socp']['trajs_data']
        fmpc_socp_inf_times = []
        for traj_inf_time in fmpc_socp_trajs['inference_time_data']:
            fmpc_socp_inf_times.extend(traj_inf_time)
        if len(fmpc_socp_inf_times) > 0:
            data_to_plot.append(np.array(fmpc_socp_inf_times) * 1000)  # Convert to ms
            labels.append('FMPC+SOCP')
            colors.append(tum_dia_dark_orange)

    # GPMPC
    if 'gpmpc' in results_dict:
        gpmpc_trajs = results_dict['gpmpc']['trajs_data']
        gpmpc_inf_times = []
        for traj_inf_time in gpmpc_trajs['inference_time_data']:
            gpmpc_inf_times.extend(traj_inf_time)
        if len(gpmpc_inf_times) > 0:
            data_to_plot.append(np.array(gpmpc_inf_times) * 1000)  # Convert to ms
            labels.append('GPMPC')
            colors.append('#DAA520')  # Gold color for GPMPC

    if len(data_to_plot) == 0:
        print("No inference time data available for plotting.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create violin plot
    parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                          showmeans=True, showmedians=True, showextrema=True)

    # Color the violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    # Style the mean, median, and extrema lines
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    parts['cbars'].set_color('black')
    parts['cmaxes'].set_color('black')
    parts['cmins'].set_color('black')

    # Add labels and formatting
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Inference Time (ms)', fontsize=12)
    ax.set_title('Controller Inference Time Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add statistics text
    for i, (data, label) in enumerate(zip(data_to_plot, labels)):
        mean_val = np.mean(data)
        median_val = np.median(data)
        ax.text(i, ax.get_ylim()[1] * 0.95,
                f'Mean: {mean_val:.2f}ms\nMedian: {median_val:.2f}ms',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'inference_time_violin.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.0)
    print(f'\nViolin plot saved to: {output_path}')

    # Also save as PNG for quick viewing
    output_path_png = os.path.join(output_dir, 'inference_time_violin.png')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    print(f'Violin plot (PNG) saved to: {output_path_png}')


def plot_tracking_error_distribution(results_dict, output_dir='./monte_carlo_results/normal', ctrl_freq=50):
    """Create plot showing tracking error distribution over time for each controller.

    Args:
        results_dict (dict): Dictionary containing results for each controller
        output_dir (str): Directory to save the plot
        ctrl_freq (int): Control frequency in Hz (for time axis)
    """
    # Define TUM colors (matching plot_hardware.py)
    tum_blue_3 = '#0073CF'
    tum_dia_dark_green = '#007C30'
    tum_dia_dark_orange = '#D64C13'

    # Figure size and settings matching plot_hardware.py
    fig_width = 7.25/4.1  # inches
    fig_height = fig_width/1.4

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Apply font settings matching plot_hardware.py
    #plt.rcParams.update({
    #    'font.size': 4,
    #    'legend.fontsize': 4,
    #})

    sample_time = 1.0 / ctrl_freq

    # Line settings matching plot_hardware.py
    alpha_lines = 0.6
    linewidth = 1.0

    # Process each controller
    controllers = []
    if 'nmpc' in results_dict:
        controllers.append(('nmpc', 'NMPC', tum_blue_3))
    if 'fmpc' in results_dict:
        controllers.append(('fmpc', 'FMPC', tum_dia_dark_green))
    if 'fmpc_socp' in results_dict:
        controllers.append(('fmpc_socp', 'FMPC+SOCP', tum_dia_dark_orange))
    if 'gpmpc' in results_dict:
        controllers.append(('gpmpc', 'GPMPC', '#DAA520'))

    for ctrl_key, ctrl_label, ctrl_color in controllers:
        trajs_data = results_dict[ctrl_key]['trajs_data']

        # Extract tracking error (MSE) from info for all trials
        all_errors = []
        max_len = 0

        for trial_info in trajs_data['info']:
            trial_errors = []
            for info_dict in trial_info:
                if 'mse' in info_dict:
                    # Take square root to get RMSE at each timestep
                    trial_errors.append(np.sqrt(info_dict['mse']))
                else:
                    trial_errors.append(0)  # Initial timestep

            all_errors.append(np.array(trial_errors))
            max_len = max(max_len, len(trial_errors))

        # Pad shorter trajectories with NaN and create matrix
        error_matrix = np.full((len(all_errors), max_len), np.nan)
        for i, errors in enumerate(all_errors):
            error_matrix[i, :len(errors)] = errors

        # Compute mean and std across trials at each timestep
        # Use nanmean/nanstd to ignore NaN values from shorter trajectories
        mean_error = np.nanmean(error_matrix, axis=0)
        std_error = np.nanstd(error_matrix, axis=0)

        # Create time axis
        time = np.arange(len(mean_error)) * sample_time

        # Keep in meters
        mean_error_m = mean_error
        std_error_m = std_error

        # Plot mean line (using hardware plot styling)
        ax.plot(time, mean_error_m, color=ctrl_color, linewidth=linewidth,
                label=ctrl_label, alpha=alpha_lines)

        # Plot shaded region for ±1 std
        ax.fill_between(time,
                        mean_error_m - std_error_m,
                        mean_error_m + std_error_m,
                        color=ctrl_color, alpha=0.2)

    # Formatting (matching plot_hardware.py)
    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(r'Tracking error (m)')
    ax.legend()
    ax.grid()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'tracking_error_distribution.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.0)
    print(f'\nTracking error distribution plot saved to: {output_path}')

    # Also save as PNG for quick viewing
    output_path_png = os.path.join(output_dir, 'tracking_error_distribution.png')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    print(f'Tracking error distribution plot (PNG) saved to: {output_path_png}')

    # Also save as PGF for LaTeX
    output_path_pgf = os.path.join(output_dir, 'tracking_error_distribution.pgf')
    plt.savefig(output_path_pgf, format='pgf', bbox_inches='tight', pad_inches=0.0)
    print(f'Tracking error distribution plot (PGF) saved to: {output_path_pgf}')


def plot_position_distribution(results_dict, output_dir='./monte_carlo_results/normal',
                               is_constrained=False, constraint_state=-0.8):
    """Create 2D position plot showing mean trajectory and ±1 std bands for each controller.

    Args:
        results_dict (dict): Dictionary containing results for each controller
        output_dir (str): Directory to save the plot
        is_constrained (bool): Whether to show constraint boundaries
        constraint_state (float): X position constraint boundary (if constrained)
    """
    # Define TUM colors (matching plot_hardware.py)
    tum_blue_3 = '#0073CF'
    tum_dia_dark_green = '#007C30'
    tum_dia_dark_orange = '#D64C13'
    tum_dia_red = '#C4071B'

    # Figure size and settings matching plot_hardware.py
    fig_width = 7.25/4.1  # inches
    fig_height = fig_width/1.4

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Apply font settings matching plot_hardware.py
    #plt.rcParams.update({
    #    'font.size': 4,
    #    'legend.fontsize': 4,
    #})

    # Line settings matching plot_hardware.py
    alpha_lines = 0.6
    alpha_constraint = 0.2
    linewidth = 1.0

    # Process each controller
    controllers = []
    if 'nmpc' in results_dict:
        controllers.append(('nmpc', 'NMPC', tum_blue_3))
    if 'fmpc' in results_dict and not is_constrained:
        # Don't show FMPC in constrained case
        controllers.append(('fmpc', 'FMPC', tum_dia_dark_green))
    if 'fmpc_socp' in results_dict:
        controllers.append(('fmpc_socp', 'FMPC+SOCP', tum_dia_dark_orange))
    if 'gpmpc' in results_dict:
        controllers.append(('gpmpc', 'GPMPC', '#DAA520'))

    # First, get reference trajectory from any controller's data
    ref_plotted = False
    for ctrl_key, _, _ in controllers:
        trajs_data = results_dict[ctrl_key]['trajs_data']
        if 'controller_data' in trajs_data and len(trajs_data['controller_data']) > 0:
            ctrl_data = trajs_data['controller_data'][0]
            if 'goal_states' in ctrl_data:
                state_ref = np.array(ctrl_data['goal_states'])

                # Debug: print actual shape
                print(f"Debug: goal_states shape for {ctrl_key}: {state_ref.shape}, ndim: {state_ref.ndim}")

                # Extract reference x and z positions based on actual shape
                try:
                    if state_ref.ndim == 4:
                        # Shape: (n_episodes, n_timesteps, n_states, T+1)
                        # Extract immediate reference (horizon index 0) from first episode
                        ref_x = state_ref[0, :, 0, 0]  # x positions
                        ref_z = state_ref[0, :, 2, 0]  # z positions
                    elif state_ref.ndim == 3:
                        # Shape: (n_timesteps, n_states, T+1)
                        # Extract immediate reference (horizon index 0)
                        ref_x = state_ref[:, 0, 0]  # x positions
                        ref_z = state_ref[:, 2, 0]  # z positions
                    elif state_ref.ndim == 2:
                        # Shape could be: (n_timesteps, n_states) OR (n_states, T+1) OR (n_episodes, n_states)
                        # For 2D quadrotor: 6 regular states or 8 flat states
                        # Check which dimension matches known state counts
                        known_state_dims = [6, 8]  # Regular state: 6, Flat state: 8

                        if state_ref.shape[1] in known_state_dims:
                            # Shape: (n_timesteps, n_states) or (n_episodes, n_states)
                            ref_x = state_ref[:, 0]  # x positions
                            ref_z = state_ref[:, 2]  # z positions
                        elif state_ref.shape[0] in known_state_dims:
                            # Shape: (n_states, T+1) or (n_states, n_timesteps) - transposed
                            ref_x = state_ref[0, :]  # x positions across time/horizon
                            ref_z = state_ref[2, :]  # z positions across time/horizon
                        else:
                            # Neither dimension matches known state counts
                            # Fall back to size heuristic: assume larger dimension is time
                            if state_ref.shape[1] > state_ref.shape[0]:
                                # Probably (n_states, n_timesteps) or (n_states, T+1)
                                if state_ref.shape[0] >= 3:  # Need at least x, x_dot, z
                                    ref_x = state_ref[0, :]
                                    ref_z = state_ref[2, :]
                                else:
                                    print(f"Warning: Cannot extract x and z from shape {state_ref.shape} (first dim too small)")
                                    continue
                            else:
                                # Probably (n_timesteps, n_states) or (n_episodes, n_states)
                                if state_ref.shape[1] >= 3:  # Need at least x, x_dot, z
                                    ref_x = state_ref[:, 0]
                                    ref_z = state_ref[:, 2]
                                else:
                                    print(f"Warning: Cannot extract x and z from shape {state_ref.shape} (second dim too small)")
                                    continue
                    elif state_ref.ndim == 1:
                        # Single state vector - can't plot trajectory
                        print(f"Warning: goal_states is 1D with shape {state_ref.shape}, cannot plot trajectory")
                        continue
                    else:
                        print(f"Warning: Unexpected goal_states ndim {state_ref.ndim} with shape {state_ref.shape}")
                        continue

                    ax.plot(ref_x, ref_z, linestyle='dashed', color='black',
                           label='_nolegend_', linewidth=linewidth, alpha=alpha_lines)
                    ref_plotted = True
                    break

                except IndexError as e:
                    print(f"Warning: Failed to extract reference from {ctrl_key} goal_states with shape {state_ref.shape}: {e}")
                    continue

    # Plot each controller's mean trajectory with std bands
    for ctrl_key, ctrl_label, ctrl_color in controllers:
        trajs_data = results_dict[ctrl_key]['trajs_data']

        # Extract x and z positions from all trials
        all_x_positions = []
        all_z_positions = []
        max_len = 0

        for trial_obs in trajs_data['obs']:
            # obs has shape (n_timesteps, n_states)
            # State: [x, x_dot, z, z_dot, theta, theta_dot]
            x_positions = trial_obs[:, 0]  # x positions
            z_positions = trial_obs[:, 2]  # z positions

            all_x_positions.append(x_positions)
            all_z_positions.append(z_positions)
            max_len = max(max_len, len(x_positions))

        # Pad shorter trajectories with NaN and create matrices
        x_matrix = np.full((len(all_x_positions), max_len), np.nan)
        z_matrix = np.full((len(all_z_positions), max_len), np.nan)

        for i, (x_pos, z_pos) in enumerate(zip(all_x_positions, all_z_positions)):
            x_matrix[i, :len(x_pos)] = x_pos
            z_matrix[i, :len(z_pos)] = z_pos

        # Compute mean and std across trials at each timestep
        mean_x = np.nanmean(x_matrix, axis=0)
        std_x = np.nanstd(x_matrix, axis=0)
        mean_z = np.nanmean(z_matrix, axis=0)
        std_z = np.nanstd(z_matrix, axis=0)

        # Create upper and lower bounds for the std band
        upper_x = mean_x + std_x
        upper_z = mean_z + std_z
        lower_x = mean_x - std_x
        lower_z = mean_z - std_z

        # Create polygon for shaded region: upper trajectory forward, lower trajectory backward
        # Remove NaN values
        valid_indices = ~(np.isnan(mean_x) | np.isnan(mean_z))

        if np.any(valid_indices):
            # Create vertices for polygon
            upper_vertices = np.column_stack([upper_x[valid_indices], upper_z[valid_indices]])
            lower_vertices = np.column_stack([lower_x[valid_indices], lower_z[valid_indices]])

            # Combine: upper forward + lower backward
            polygon_vertices = np.vstack([upper_vertices, lower_vertices[::-1]])

            # Plot shaded region
            ax.fill(polygon_vertices[:, 0], polygon_vertices[:, 1],
                   color=ctrl_color, alpha=0.2)

            # Plot mean trajectory (using hardware plot styling)
            ax.plot(mean_x[valid_indices], mean_z[valid_indices],
                   color=ctrl_color, linewidth=linewidth, label=ctrl_label, alpha=alpha_lines)

    # Add constraint boundary if in constrained mode
    if is_constrained:
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        # Shade the region x < constraint_state
        ax.axvspan(xlim[0], constraint_state, color=tum_dia_red, alpha=alpha_constraint)

    # Formatting (matching plot_hardware.py)
    ax.set_xlabel(r'Position x (m)')
    ax.set_ylabel(r'Position z (m)')
    ax.legend()
    ax.grid()

    # Set reasonable axis limits
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([0.4, 1.6])

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'position_distribution.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.0)
    print(f'\nPosition distribution plot saved to: {output_path}')

    # Also save as PNG for quick viewing
    output_path_png = os.path.join(output_dir, 'position_distribution.png')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    print(f'Position distribution plot (PNG) saved to: {output_path_png}')

    # Also save as PGF for LaTeX
    output_path_pgf = os.path.join(output_dir, 'position_distribution.pgf')
    plt.savefig(output_path_pgf, format='pgf', bbox_inches='tight', pad_inches=0.0)
    print(f'Position distribution plot (PGF) saved to: {output_path_pgf}')


def plot_input_distribution(results_dict, output_dir='./monte_carlo_results/normal',
                           ctrl_freq=50, is_constrained=False, constraint_input=0.435):
    """Create plot showing input distribution over time for each controller.

    Args:
        results_dict (dict): Dictionary containing results for each controller
        output_dir (str): Directory to save the plot
        ctrl_freq (int): Control frequency in Hz (for time axis)
        is_constrained (bool): Whether to show constraint boundaries
        constraint_input (float): Thrust constraint boundary (if constrained)
    """
    # Define TUM colors (matching plot_hardware.py)
    tum_blue_3 = '#0073CF'
    tum_dia_dark_green = '#007C30'
    tum_dia_dark_orange = '#D64C13'
    tum_dia_red = '#C4071B'

    # Figure size: 3 inches width (as requested)
    fig_width = 3.0  # inches
    fig_height = fig_width/1.6

    fig, ax = plt.subplots(2, 1, figsize=(fig_width, fig_height))

    # Note: Font settings are already set globally at the top of this module
    # No need to override them here

    # Line settings matching plot_hardware.py
    alpha_lines = 0.6
    alpha_constraint = 0.2
    linewidth = 1.0

    sample_time = 1.0 / ctrl_freq

    # Process each controller
    controllers = []
    if 'nmpc' in results_dict:
        controllers.append(('nmpc', 'NMPC', tum_blue_3))
    if 'fmpc' in results_dict and not is_constrained:
        # Don't show FMPC in constrained case
        controllers.append(('fmpc', 'FMPC', tum_dia_dark_green))
    if 'fmpc_socp' in results_dict:
        controllers.append(('fmpc_socp', 'FMPC+SOCP', tum_dia_dark_orange))
    if 'gpmpc' in results_dict:
        controllers.append(('gpmpc', 'GPMPC', '#DAA520'))

    for ctrl_key, ctrl_label, ctrl_color in controllers:
        trajs_data = results_dict[ctrl_key]['trajs_data']

        # Extract actions (inputs) from all trials
        all_thrusts = []
        all_angles = []
        max_len = 0

        for trial_action in trajs_data['action']:
            # action has shape (n_timesteps, 2) where 2 is [thrust, angle]
            thrusts = trial_action[:, 0]  # Thrust T_c
            angles = trial_action[:, 1]   # Angle theta_c

            all_thrusts.append(thrusts)
            all_angles.append(angles)
            max_len = max(max_len, len(thrusts))

        # Pad shorter trajectories with NaN and create matrices
        thrust_matrix = np.full((len(all_thrusts), max_len), np.nan)
        angle_matrix = np.full((len(all_angles), max_len), np.nan)

        for i, (thrust, angle) in enumerate(zip(all_thrusts, all_angles)):
            thrust_matrix[i, :len(thrust)] = thrust
            angle_matrix[i, :len(angle)] = angle

        # Compute mean and std across trials at each timestep
        mean_thrust = np.nanmean(thrust_matrix, axis=0)
        std_thrust = np.nanstd(thrust_matrix, axis=0)
        mean_angle = np.nanmean(angle_matrix, axis=0)
        std_angle = np.nanstd(angle_matrix, axis=0)

        # Create time axis
        time = np.arange(len(mean_thrust)) * sample_time

        # Plot thrust (top subplot) - using hardware plot styling
        ax[0].plot(time, mean_thrust, color=ctrl_color, linewidth=linewidth,
                   label=ctrl_label, alpha=alpha_lines)
        ax[0].fill_between(time,
                          mean_thrust - std_thrust,
                          mean_thrust + std_thrust,
                          color=ctrl_color, alpha=0.2)

        # Plot angle (bottom subplot) - using hardware plot styling
        ax[1].plot(time, mean_angle, color=ctrl_color, linewidth=linewidth,
                   label=ctrl_label, alpha=alpha_lines)
        ax[1].fill_between(time,
                          mean_angle - std_angle,
                          mean_angle + std_angle,
                          color=ctrl_color, alpha=0.2)

    # Add constraint boundary if in constrained mode
    if is_constrained:
        # Shade the region above constraint_input for thrust
        ylim_top = ax[0].get_ylim()
        ax[0].axhspan(constraint_input, ylim_top[1], color=tum_dia_red, alpha=alpha_constraint)

        # Shade the regions outside ±1.5 rad for angle
        ylim_bottom = ax[1].get_ylim()
        ax[1].axhspan(ylim_bottom[0], -1.5, color=tum_dia_red, alpha=alpha_constraint)
        ax[1].axhspan(1.5, ylim_bottom[1], color=tum_dia_red, alpha=alpha_constraint)

    # Formatting (matching plot_hardware.py)
    ax[0].set_ylabel(r'Thrust $T_c$ (N)')
    ax[0].grid()

    ax[1].set_xlabel(r'Time (s)')
    ax[1].set_ylabel(r'Angle $\theta_c$ (rad)')
    ax[1].legend(loc='upper right')
    ax[1].grid()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'input_distribution.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f'\nInput distribution plot saved to: {output_path}')

    # Also save as PNG for quick viewing
    output_path_png = os.path.join(output_dir, 'input_distribution.png')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f'Input distribution plot (PNG) saved to: {output_path_png}')

    # Also save as PGF for LaTeX
    output_path_pgf = os.path.join(output_dir, 'input_distribution.pgf')
    plt.savefig(output_path_pgf, format='pgf', bbox_inches='tight')
    print(f'Input distribution plot (PGF) saved to: {output_path_pgf}')
