#!/usr/bin/env python3
"""Monte Carlo experiment runner for comparing MPC controllers with randomized initial conditions.

This script runs multiple trials of the controllers (NMPC, FMPC, FMPC+SOCP) with randomized
initial conditions to gather statistically significant performance metrics.
"""

import os
import pickle
import argparse
from functools import partial
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import sys
import traceback

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def generate_initial_states(env_func, n_trials, base_seed):
    """Generate a set of randomized initial states for Monte Carlo trials.

    Args:
        env_func: Function to create the environment
        n_trials (int): Number of initial states to generate
        base_seed (int): Base seed for reproducibility

    Returns:
        list: List of initial states (as numpy arrays)
        list: List of seeds used to generate each state
    """
    print(f'\nGenerating {n_trials} randomized initial states...')

    # Create environment with randomization enabled
    random_env = env_func(gui=False)

    initial_states = []
    seeds = []

    for i in range(n_trials):
        seed = base_seed + i
        seeds.append(seed)

        # Reset with specific seed to get randomized initial state
        obs, _ = random_env.reset(seed=seed)

        # Get the actual state (not just observation)
        init_state = random_env.state.copy()
        initial_states.append(init_state)

        print(f'  Trial {i+1}/{n_trials}: seed={seed}, state={init_state}')

    random_env.close()

    print(f'Generated {len(initial_states)} initial states.\n')
    return initial_states, seeds


def run_controller_trials(controller_name, env_func, ctrl_func, initial_states, gui=False):
    """Run a controller on all initial states.

    Args:
        controller_name (str): Name of the controller (for logging)
        env_func: Function to create the environment
        ctrl_func: Function to create the controller
        initial_states (list): List of initial states to test
        gui (bool): Whether to show GUI

    Returns:
        dict: Aggregated trajectory data from successful trials
        dict: Aggregated metrics from successful trials
        list: List of failed trial information with partial trajectories
    """
    from copy import deepcopy

    n_trials = len(initial_states)
    print(f'\n{"="*80}')
    print(f'Running {controller_name} on {n_trials} trials...')
    print(f'{"="*80}\n')

    successful_trajs = defaultdict(list)
    failed_runs = []

    for trial_idx, init_state in enumerate(initial_states):
        print(f'  Trial {trial_idx+1}/{n_trials} - init_state: {init_state}')

        env = None
        train_env = None
        ctrl = None
        experiment = None

        try:
            # Create environment with fixed (non-randomized) initial state
            env = env_func(gui=gui, randomized_init=False, init_state=init_state)
            train_env = env_func(gui=False, randomized_init=False, init_state=init_state)

            # Create controller
            ctrl = ctrl_func()

            # Create experiment
            experiment = BaseExperiment(env=env, ctrl=ctrl, train_env=train_env)
            experiment.launch_training()

            # Run evaluation for this trial (1 episode)
            trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1, verbose=False)

            # Success: collect trajectory data
            for key, value in trajs_data.items():
                successful_trajs[key] += value

            print(f'    Trial {trial_idx+1} completed.\n')

        except Exception as e:
            # Failure: extract partial trajectory data
            print(f'    Trial {trial_idx+1} FAILED: {type(e).__name__}')
            traceback.print_exc()

            partial_trajs = None
            timestep_failed = 0

            try:
                # Try to get partial trajectory from controller
                if experiment is not None and hasattr(experiment, 'ctrl') and hasattr(experiment.ctrl, 'results_dict'):
                    ctrl_data = experiment.ctrl.results_dict

                    # Check if there's actual data
                    if ctrl_data and isinstance(ctrl_data, dict):
                        obs_list = ctrl_data.get('obs', [])
                        if len(obs_list) > 0:
                            timestep_failed = len(obs_list)
                            # Deep copy to preserve data
                            partial_trajs = {}
                            for key, value in ctrl_data.items():
                                try:
                                    partial_trajs[key] = deepcopy(value)
                                except:
                                    # Some objects can't be deep copied
                                    try:
                                        partial_trajs[key] = value.copy() if hasattr(value, 'copy') else value
                                    except:
                                        partial_trajs[key] = None
            except Exception as extract_error:
                print(f'    Warning: Could not extract partial trajectory: {extract_error}')
                partial_trajs = None

            failure_info = {
                'trial_idx': trial_idx,
                'init_state': init_state.copy(),
                'error_type': type(e).__name__,
                'error_msg': str(e)[:500],  # Truncate long error messages
                'timestep_failed': timestep_failed,
                'partial_trajectory': partial_trajs,
                'has_partial_data': partial_trajs is not None
            }
            failed_runs.append(failure_info)

            if timestep_failed > 0:
                print(f'    Partial trajectory saved ({timestep_failed} timesteps)\n')
            else:
                print(f'    No partial trajectory data available\n')

        finally:
            # Clean up resources (important even on failure!)
            try:
                if env is not None:
                    env.close()
            except:
                pass
            try:
                if train_env is not None:
                    train_env.close()
            except:
                pass
            try:
                if ctrl is not None:
                    ctrl.close()
            except:
                pass

    # Compute aggregated metrics across successful trials only
    n_successful = len(successful_trajs.get('obs', []))
    n_failed = len(failed_runs)

    print(f'Computing aggregated metrics for {controller_name}...')
    print(f'  Successful trials: {n_successful}')
    print(f'  Failed trials: {n_failed}')

    if n_successful > 0:
        successful_trajs = dict(successful_trajs)

        # Use the experiment's metric extractor to compute statistics
        from safe_control_gym.experiments.base_experiment import MetricExtractor
        metric_extractor = MetricExtractor()

        # We need a MAX_STEPS value - get it from the environment
        temp_env = env_func(gui=False)
        MAX_STEPS = int(temp_env.CTRL_FREQ * temp_env.EPISODE_LEN_SEC)
        temp_env.close()

        metrics = metric_extractor.compute_metrics(data=successful_trajs, max_steps=MAX_STEPS, verbose=False)

        # Add success/failure statistics
        metrics['n_trials'] = n_trials
        metrics['n_successful'] = n_successful
        metrics['n_failed'] = n_failed
        metrics['success_rate'] = n_successful / n_trials if n_trials > 0 else 0

        # Compute inference time std dev
        metrics['inference_time_std'] = np.std(metrics['avarage_inference_time'])

        print(f'{controller_name} completed:')
        print(f'  Average RMSE: {metrics["average_rmse"]:.4f} ± {metrics["rmse_std"]:.4f}')
        print(f'  Average inference time: {np.mean(metrics["avarage_inference_time"]):.4f}s ± {metrics["inference_time_std"]:.4f}s')
        print(f'  Success rate: {metrics["success_rate"]:.2%}')
        print()
    else:
        # No successful trials
        print(f'{controller_name}: All trials failed!')
        metrics = {
            'n_trials': n_trials,
            'n_successful': 0,
            'n_failed': n_failed,
            'success_rate': 0.0,
            'average_rmse': 0.0,
            'rmse_std': 0.0,
            'failure_rate': 0.0,
            'avarage_inference_time': [0.0],
            'inference_time_std': 0.0
        }
        successful_trajs = {}
        print()

    return successful_trajs, metrics, failed_runs


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

    ctrl_data = trajs_data['controller_data'][0]

    # MPC solve time (available for all controllers)
    if 'mpc_solve_time' in ctrl_data:
        all_mpc_times = []
        for episode_data in ctrl_data['mpc_solve_time']:
            all_mpc_times.extend(episode_data)
        if len(all_mpc_times) > 0:
            timing_stats['mpc_solve'] = {
                'mean': np.mean(all_mpc_times) * 1000,  # Convert to ms
                'std': np.std(all_mpc_times) * 1000
            }

    # FMPC_SOCP specific timings
    if controller_name.lower() == 'fmpc_socp' or controller_name.lower() == 'fmpc+socp':
        # SOCP solve time
        if 'socp_solve_time' in ctrl_data:
            all_socp_times = []
            for episode_data in ctrl_data['socp_solve_time']:
                all_socp_times.extend(episode_data)
            if len(all_socp_times) > 0:
                timing_stats['socp_solve'] = {
                    'mean': np.mean(all_socp_times) * 1000,
                    'std': np.std(all_socp_times) * 1000
                }

        # GP inference time (list of 2 values per timestep)
        if 'gp_time' in ctrl_data:
            all_gp_times = []
            for episode_data in ctrl_data['gp_time']:
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
        if 'observer_time' in ctrl_data:
            all_observer_times = []
            for episode_data in ctrl_data['observer_time']:
                all_observer_times.extend(episode_data)
            if len(all_observer_times) > 0:
                timing_stats['observer'] = {
                    'mean': np.mean(all_observer_times) * 1000,
                    'std': np.std(all_observer_times) * 1000
                }

        # Flat transformation time
        if 'flat_transform_time' in ctrl_data:
            all_transform_times = []
            for episode_data in ctrl_data['flat_transform_time']:
                all_transform_times.extend(episode_data)
            if len(all_transform_times) > 0:
                timing_stats['flat_transform'] = {
                    'mean': np.mean(all_transform_times) * 1000,
                    'std': np.std(all_transform_times) * 1000
                }

        # Dynamic extension time
        if 'dyn_ext_time' in ctrl_data:
            all_dyn_ext_times = []
            for episode_data in ctrl_data['dyn_ext_time']:
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

    # Print table
    print(f'\n{"Component":<25} | {"NMPC":>20} | {"FMPC":>20} | {"FMPC+SOCP":>20}')
    print('-'*90)

    # MPC solve time
    nmpc_mpc = nmpc_timing.get('mpc_solve', {'mean': 0, 'std': 0})
    fmpc_mpc = fmpc_timing.get('mpc_solve', {'mean': 0, 'std': 0})
    fmpc_socp_mpc = fmpc_socp_timing.get('mpc_solve', {'mean': 0, 'std': 0})

    print(f'{"MPC Solve":<25} | {nmpc_mpc["mean"]:>8.3f} ± {nmpc_mpc["std"]:>7.3f} | '
          f'{fmpc_mpc["mean"]:>8.3f} ± {fmpc_mpc["std"]:>7.3f} | '
          f'{fmpc_socp_mpc["mean"]:>8.3f} ± {fmpc_socp_mpc["std"]:>7.3f}')

    # SOCP solve time (only for FMPC+SOCP)
    if 'socp_solve' in fmpc_socp_timing:
        socp = fmpc_socp_timing['socp_solve']
        print(f'{"SOCP Solve":<25} | {"—":>20} | {"—":>20} | '
              f'{socp["mean"]:>8.3f} ± {socp["std"]:>7.3f}')

    # GP inference time (only for FMPC+SOCP)
    if 'gp_inference' in fmpc_socp_timing:
        gp = fmpc_socp_timing['gp_inference']
        print(f'{"GP Inference":<25} | {"—":>20} | {"—":>20} | '
              f'{gp["mean"]:>8.3f} ± {gp["std"]:>7.3f}')

    # Observer time (only for FMPC+SOCP)
    if 'observer' in fmpc_socp_timing:
        obs = fmpc_socp_timing['observer']
        print(f'{"Flat State Observer":<25} | {"—":>20} | {"—":>20} | '
              f'{obs["mean"]:>8.3f} ± {obs["std"]:>7.3f}')

    # Flat transformation time (only for FMPC+SOCP)
    if 'flat_transform' in fmpc_socp_timing:
        ft = fmpc_socp_timing['flat_transform']
        print(f'{"Flat Transformation":<25} | {"—":>20} | {"—":>20} | '
              f'{ft["mean"]:>8.3f} ± {ft["std"]:>7.3f}')

    # Dynamic extension time (only for FMPC+SOCP)
    if 'dyn_extension' in fmpc_socp_timing:
        de = fmpc_socp_timing['dyn_extension']
        print(f'{"Dynamic Extension":<25} | {"—":>20} | {"—":>20} | '
              f'{de["mean"]:>8.3f} ± {de["std"]:>7.3f}')

    print('-'*90)
    print()


def save_results(output_dir, initial_states, seeds, results_dict):
    """Save all results to disk.

    Args:
        output_dir (str): Directory to save results
        initial_states (list): List of initial states used
        seeds (list): List of seeds used
        results_dict (dict): Dictionary containing results for each controller
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save initial states and seeds
    with open(os.path.join(output_dir, 'initial_states.pkl'), 'wb') as f:
        pickle.dump({'initial_states': initial_states, 'seeds': seeds}, f)
    print(f'Saved initial states to {output_dir}/initial_states.pkl')

    # Save each controller's results
    for controller_name, data in results_dict.items():
        # Save successful trials with trajectory data and metrics
        filename = f'{controller_name}_trials.pkl'
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f'Saved {controller_name} results to {filepath}')

        # Save failed trials separately if there are any
        if 'failed_runs' in data and len(data['failed_runs']) > 0:
            failed_filename = f'{controller_name}_failed_trials.pkl'
            failed_filepath = os.path.join(output_dir, failed_filename)
            with open(failed_filepath, 'wb') as f:
                pickle.dump(data['failed_runs'], f)
            print(f'Saved {controller_name} failed trials to {failed_filepath} '
                  f'({len(data["failed_runs"])} failures)')

    # Save aggregated metrics summary
    metrics_summary = {}
    for controller_name, data in results_dict.items():
        metrics_summary[controller_name] = data['metrics']

    with open(os.path.join(output_dir, 'aggregated_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics_summary, f)
    print(f'Saved aggregated metrics to {output_dir}/aggregated_metrics.pkl')


def print_summary_table(results_dict):
    """Print a summary table comparing all controllers.

    Args:
        results_dict (dict): Dictionary containing results for each controller
    """
    print('\n' + '='*80)
    print('SUMMARY: Monte Carlo Experiment Results')
    print('='*80)

    # Extract metrics
    nmpc_metrics = results_dict.get('nmpc', {}).get('metrics', {})
    fmpc_metrics = results_dict.get('fmpc', {}).get('metrics', {})
    fmpc_socp_metrics = results_dict.get('fmpc_socp', {}).get('metrics', {})

    # Print trial statistics first
    print('\n' + '='*80)
    print('TRIAL STATISTICS')
    print('='*80)
    print(f'\n{"Statistic":<30} | {"NMPC":>15} | {"FMPC":>15} | {"FMPC+SOCP":>15}')
    print('-'*80)

    # Number of trials
    nmpc_n_trials = nmpc_metrics.get('n_trials', 0)
    fmpc_n_trials = fmpc_metrics.get('n_trials', 0)
    fmpc_socp_n_trials = fmpc_socp_metrics.get('n_trials', 0)

    print(f'{"Total Trials":<30} | {nmpc_n_trials:>15d} | '
          f'{fmpc_n_trials:>15d} | '
          f'{fmpc_socp_n_trials:>15d}')

    # Successful trials
    nmpc_n_success = nmpc_metrics.get('n_successful', 0)
    fmpc_n_success = fmpc_metrics.get('n_successful', 0)
    fmpc_socp_n_success = fmpc_socp_metrics.get('n_successful', 0)

    print(f'{"Successful Trials":<30} | {nmpc_n_success:>15d} | '
          f'{fmpc_n_success:>15d} | '
          f'{fmpc_socp_n_success:>15d}')

    # Failed trials
    nmpc_n_failed = nmpc_metrics.get('n_failed', 0)
    fmpc_n_failed = fmpc_metrics.get('n_failed', 0)
    fmpc_socp_n_failed = fmpc_socp_metrics.get('n_failed', 0)

    print(f'{"Failed Trials":<30} | {nmpc_n_failed:>15d} | '
          f'{fmpc_n_failed:>15d} | '
          f'{fmpc_socp_n_failed:>15d}')

    # Success rate
    nmpc_success_rate = nmpc_metrics.get('success_rate', 0)
    fmpc_success_rate = fmpc_metrics.get('success_rate', 0)
    fmpc_socp_success_rate = fmpc_socp_metrics.get('success_rate', 0)

    print(f'{"Success Rate (%)":<30} | {nmpc_success_rate*100:>15.1f} | '
          f'{fmpc_success_rate*100:>15.1f} | '
          f'{fmpc_socp_success_rate*100:>15.1f}')

    print('-'*80)

    # Performance metrics (computed over successful trials only)
    print('\n' + '='*80)
    print('PERFORMANCE METRICS (Successful Trials Only)')
    print('='*80)
    print(f'\n{"Metric":<30} | {"NMPC":>15} | {"FMPC":>15} | {"FMPC+SOCP":>15}')
    print('-'*80)

    # Full trajectory RMSE
    print(f'{"Average RMSE (m)":<30} | {nmpc_metrics.get("average_rmse", 0):>15.4f} | '
          f'{fmpc_metrics.get("average_rmse", 0):>15.4f} | '
          f'{fmpc_socp_metrics.get("average_rmse", 0):>15.4f}')

    print(f'{"RMSE Std Dev (m)":<30} | {nmpc_metrics.get("rmse_std", 0):>15.4f} | '
          f'{fmpc_metrics.get("rmse_std", 0):>15.4f} | '
          f'{fmpc_socp_metrics.get("rmse_std", 0):>15.4f}')

    # Second half RMSE
    nmpc_second_half = (0.0, 0.0)
    fmpc_second_half = (0.0, 0.0)
    fmpc_socp_second_half = (0.0, 0.0)

    if 'nmpc' in results_dict:
        nmpc_second_half = compute_second_half_rmse(results_dict['nmpc']['trajs_data'])[:2]
    if 'fmpc' in results_dict:
        fmpc_second_half = compute_second_half_rmse(results_dict['fmpc']['trajs_data'])[:2]
    if 'fmpc_socp' in results_dict:
        fmpc_socp_second_half = compute_second_half_rmse(results_dict['fmpc_socp']['trajs_data'])[:2]

    print(f'{"RMSE Second Half (m)":<30} | {nmpc_second_half[0]:>15.4f} | '
          f'{fmpc_second_half[0]:>15.4f} | '
          f'{fmpc_socp_second_half[0]:>15.4f}')

    print(f'{"RMSE Second Half Std (m)":<30} | {nmpc_second_half[1]:>15.4f} | '
          f'{fmpc_second_half[1]:>15.4f} | '
          f'{fmpc_socp_second_half[1]:>15.4f}')

    # Inference time
    nmpc_time = np.mean(nmpc_metrics.get('avarage_inference_time', [0]))
    nmpc_time_std = nmpc_metrics.get('inference_time_std', 0)
    fmpc_time = np.mean(fmpc_metrics.get('avarage_inference_time', [0]))
    fmpc_time_std = fmpc_metrics.get('inference_time_std', 0)
    fmpc_socp_time = np.mean(fmpc_socp_metrics.get('avarage_inference_time', [0]))
    fmpc_socp_time_std = fmpc_socp_metrics.get('inference_time_std', 0)

    print(f'{"Avg Inference Time (ms)":<30} | {nmpc_time*1000:>7.2f} ± {nmpc_time_std*1000:>5.2f} | '
          f'{fmpc_time*1000:>7.2f} ± {fmpc_time_std*1000:>5.2f} | '
          f'{fmpc_socp_time*1000:>7.2f} ± {fmpc_socp_time_std*1000:>5.2f}')

    # Constraint violations
    print(f'{"Failure Rate (%)":<30} | {nmpc_metrics.get("failure_rate", 0)*100:>15.2f} | '
          f'{fmpc_metrics.get("failure_rate", 0)*100:>15.2f} | '
          f'{fmpc_socp_metrics.get("failure_rate", 0)*100:>15.2f}')

    print(f'{"Avg Constraint Violations":<30} | {nmpc_metrics.get("average_constraint_violation", 0):>15.2f} | '
          f'{fmpc_metrics.get("average_constraint_violation", 0):>15.2f} | '
          f'{fmpc_socp_metrics.get("average_constraint_violation", 0):>15.2f}')

    print('-'*80)
    print()

    # Print detailed timing breakdown
    print_timing_breakdown_table(results_dict)


def plot_inference_time_violin(results_dict, output_dir='./monte_carlo_results/normal'):
    """Create violin plot comparing inference times across controllers.

    Args:
        results_dict (dict): Dictionary containing results for each controller
        output_dir (str): Directory to save the plot
    """
    import matplotlib.pyplot as plt

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
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f'\nViolin plot saved to: {output_path}')

    # Also save as PNG for quick viewing
    output_path_png = os.path.join(output_dir, 'inference_time_violin.png')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f'Violin plot (PNG) saved to: {output_path_png}')


def plot_tracking_error_distribution(results_dict, output_dir='./monte_carlo_results/normal', ctrl_freq=50):
    """Create plot showing tracking error distribution over time for each controller.

    Args:
        results_dict (dict): Dictionary containing results for each controller
        output_dir (str): Directory to save the plot
        ctrl_freq (int): Control frequency in Hz (for time axis)
    """
    import matplotlib.pyplot as plt

    # Define TUM colors
    tum_blue_3 = '#0073CF'
    tum_dia_dark_green = '#007C30'
    tum_dia_dark_orange = '#D64C13'

    fig, ax = plt.subplots(figsize=(10, 6))

    sample_time = 1.0 / ctrl_freq

    # Process each controller
    controllers = []
    if 'nmpc' in results_dict:
        controllers.append(('nmpc', 'NMPC', tum_blue_3))
    if 'fmpc' in results_dict:
        controllers.append(('fmpc', 'FMPC', tum_dia_dark_green))
    if 'fmpc_socp' in results_dict:
        controllers.append(('fmpc_socp', 'FMPC+SOCP', tum_dia_dark_orange))

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

        # Convert to millimeters for better readability
        mean_error_mm = mean_error * 1000
        std_error_mm = std_error * 1000

        # Plot mean line
        ax.plot(time, mean_error_mm, color=ctrl_color, linewidth=2.5,
                label=ctrl_label, alpha=0.9)

        # Plot shaded region for ±1 std
        ax.fill_between(time,
                        mean_error_mm - std_error_mm,
                        mean_error_mm + std_error_mm,
                        color=ctrl_color, alpha=0.2)

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Tracking Error (mm)', fontsize=12)
    ax.set_title('Tracking Error Distribution Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'tracking_error_distribution.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f'\nTracking error distribution plot saved to: {output_path}')

    # Also save as PNG for quick viewing
    output_path_png = os.path.join(output_dir, 'tracking_error_distribution.png')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f'Tracking error distribution plot (PNG) saved to: {output_path_png}')


def plot_position_distribution(results_dict, output_dir='./monte_carlo_results/normal',
                               is_constrained=False, constraint_state=-0.8):
    """Create 2D position plot showing mean trajectory and ±1 std bands for each controller.

    Args:
        results_dict (dict): Dictionary containing results for each controller
        output_dir (str): Directory to save the plot
        is_constrained (bool): Whether to show constraint boundaries
        constraint_state (float): X position constraint boundary (if constrained)
    """
    import matplotlib.pyplot as plt

    # Define TUM colors
    tum_blue_3 = '#0073CF'
    tum_dia_dark_green = '#007C30'
    tum_dia_dark_orange = '#D64C13'
    tum_dia_red = '#C4071B'

    fig, ax = plt.subplots(figsize=(10, 7))

    # Process each controller
    controllers = []
    if 'nmpc' in results_dict:
        controllers.append(('nmpc', 'NMPC', tum_blue_3))
    if 'fmpc' in results_dict and not is_constrained:
        # Don't show FMPC in constrained case
        controllers.append(('fmpc', 'FMPC', tum_dia_dark_green))
    if 'fmpc_socp' in results_dict:
        controllers.append(('fmpc_socp', 'FMPC+SOCP', tum_dia_dark_orange))

    # First, get reference trajectory from any controller's data
    ref_plotted = False
    for ctrl_key, _, _ in controllers:
        trajs_data = results_dict[ctrl_key]['trajs_data']
        if 'controller_data' in trajs_data and len(trajs_data['controller_data']) > 0:
            ctrl_data = trajs_data['controller_data'][0]
            if 'goal_states' in ctrl_data:
                state_ref = np.array(ctrl_data['goal_states'])
                # state_ref has shape (n_episodes, n_timesteps, n_states, 1)
                # Plot first episode's reference (they should all be the same)
                ref_x = state_ref[0, :, 0, 0]  # x positions
                ref_z = state_ref[0, :, 2, 0]  # z positions
                ax.plot(ref_x, ref_z, linestyle='dashed', color='black',
                       label='Reference', linewidth=2.5, alpha=0.6)
                ref_plotted = True
                break

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

            # Plot mean trajectory
            ax.plot(mean_x[valid_indices], mean_z[valid_indices],
                   color=ctrl_color, linewidth=2.5, label=ctrl_label, alpha=0.9)

    # Add constraint boundary if in constrained mode
    if is_constrained:
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        # Shade the region x < constraint_state
        ax.axvspan(xlim[0], constraint_state, color=tum_dia_red, alpha=0.2)

    # Formatting
    ax.set_xlabel('Position x (m)', fontsize=12)
    ax.set_ylabel('Position z (m)', fontsize=12)
    ax.set_title('Position Distribution (Mean ± 1 Std)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)

    # Set reasonable axis limits
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([0.4, 1.6])

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'position_distribution.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f'\nPosition distribution plot saved to: {output_path}')

    # Also save as PNG for quick viewing
    output_path_png = os.path.join(output_dir, 'position_distribution.png')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f'Position distribution plot (PNG) saved to: {output_path_png}')


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
    import matplotlib.pyplot as plt

    # Define TUM colors
    tum_blue_3 = '#0073CF'
    tum_dia_dark_green = '#007C30'
    tum_dia_dark_orange = '#D64C13'
    tum_dia_red = '#C4071B'

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

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

        # Plot thrust (top subplot)
        ax[0].plot(time, mean_thrust, color=ctrl_color, linewidth=2.5,
                   label=ctrl_label, alpha=0.9)
        ax[0].fill_between(time,
                          mean_thrust - std_thrust,
                          mean_thrust + std_thrust,
                          color=ctrl_color, alpha=0.2)

        # Plot angle (bottom subplot)
        ax[1].plot(time, mean_angle, color=ctrl_color, linewidth=2.5,
                   label=ctrl_label, alpha=0.9)
        ax[1].fill_between(time,
                          mean_angle - std_angle,
                          mean_angle + std_angle,
                          color=ctrl_color, alpha=0.2)

    # Add constraint boundary if in constrained mode
    if is_constrained:
        # Shade the region above constraint_input for thrust
        ylim_top = ax[0].get_ylim()
        ax[0].axhspan(constraint_input, ylim_top[1], color=tum_dia_red, alpha=0.2)

        # Shade the regions outside ±1.5 rad for angle
        ylim_bottom = ax[1].get_ylim()
        ax[1].axhspan(ylim_bottom[0], -1.5, color=tum_dia_red, alpha=0.2)
        ax[1].axhspan(1.5, ylim_bottom[1], color=tum_dia_red, alpha=0.2)

    # Formatting for thrust subplot
    ax[0].set_ylabel(r'Thrust $T_c$ (N)', fontsize=12)
    ax[0].set_title('Input Distribution Over Time (Mean ± 1 Std)', fontsize=14, fontweight='bold')
    ax[0].grid(alpha=0.3)

    # Formatting for angle subplot
    ax[1].set_xlabel('Time (s)', fontsize=12)
    ax[1].set_ylabel(r'Angle $\theta_c$ (rad)', fontsize=12)
    ax[1].legend(fontsize=11, loc='upper right')
    ax[1].grid(alpha=0.3)

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'input_distribution.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f'\nInput distribution plot saved to: {output_path}')

    # Also save as PNG for quick viewing
    output_path_png = os.path.join(output_dir, 'input_distribution.png')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f'Input distribution plot (PNG) saved to: {output_path_png}')


def run_monte_carlo_experiment(mode='normal', n_trials=2, base_seed=42, gui=False,
                               run_nmpc=True, run_fmpc=True, run_fmpc_socp=True):
    """Main function to run Monte Carlo experiments.

    Args:
        mode (str): 'normal' or 'constrained'
        n_trials (int): Number of Monte Carlo trials
        base_seed (int): Base seed for reproducibility
        gui (bool): Whether to show GUI
        run_nmpc (bool): Whether to run NMPC controller
        run_fmpc (bool): Whether to run FMPC controller
        run_fmpc_socp (bool): Whether to run FMPC+SOCP controller
    """
    print('\n' + '='*80)
    print(f'Monte Carlo Experiment: {mode.upper()} mode, {n_trials} trials')
    print('='*80 + '\n')

    # Configure file paths based on mode
    if mode == 'normal':
        yaml_file_base = './config_overrides_fast/quadrotor_2D_attitude_tracking.yaml'
        yaml_file_base_random = './config_overrides_fast_random/quadrotor_2D_attitude_tracking.yaml'
        yaml_file_nmpc = './config_overrides_fast/mpc_quadrotor_2D_attitude_tracking.yaml'
        yaml_file_fmpc = './config_overrides_fast/fmpc_ext_quadrotor_2D_attitude_tracking.yaml'
        yaml_file_fmpc_socp = './config_overrides_fast/fmpc_socp_quadrotor_2D_attitude_tracking.yaml'
        output_dir = './monte_carlo_results/normal'
    elif mode == 'constrained':
        yaml_file_base = './config_overrides_constrained/quadrotor_2D_attitude_tracking.yaml'
        yaml_file_base_random = './config_overrides_constrained_random/quadrotor_2D_attitude_tracking.yaml'
        yaml_file_nmpc = './config_overrides_constrained/mpc_quadrotor_2D_attitude_tracking.yaml'
        yaml_file_fmpc = './config_overrides_constrained/fmpc_ext_quadrotor_2D_attitude_tracking.yaml'
        yaml_file_fmpc_socp = './config_overrides_constrained/fmpc_socp_quadrotor_2D_attitude_tracking.yaml'
        output_dir = './monte_carlo_results/constrained'
    else:
        raise ValueError(f'Unknown mode: {mode}')

    # Verify config files exist
    for yaml_file in [yaml_file_base, yaml_file_base_random, yaml_file_nmpc,
                      yaml_file_fmpc, yaml_file_fmpc_socp]:
        assert os.path.exists(yaml_file), f'{yaml_file} does not exist'

    # Helper function to create environment with randomization for initial state generation
    def create_random_env_func(**kwargs):
        sys.argv[1:] = ['--algo', 'mpc',
                        '--task', 'quadrotor',
                        '--overrides', yaml_file_base, yaml_file_base_random]
        CONFIG_FACTORY = ConfigFactory()
        config = CONFIG_FACTORY.merge()
        return make(config.task, **{**config.task_config, **kwargs})

    # Generate initial states
    initial_states, seeds = generate_initial_states(
        env_func=create_random_env_func,
        n_trials=n_trials,
        base_seed=base_seed
    )

    results_dict = {}

    # Run NMPC
    if run_nmpc:
        sys.argv[1:] = ['--algo', 'mpc',
                        '--task', 'quadrotor',
                        '--overrides', yaml_file_base, yaml_file_nmpc]
        CONFIG_FACTORY = ConfigFactory()
        config = CONFIG_FACTORY.merge()

        env_func = partial(make, config.task, **config.task_config)
        ctrl_func = partial(make, config.algo, env_func, **config.algo_config)

        trajs, metrics, failed_runs = run_controller_trials('NMPC', env_func, ctrl_func, initial_states, gui)
        results_dict['nmpc'] = {'trajs_data': trajs, 'metrics': metrics, 'failed_runs': failed_runs}

    # Run FMPC
    if run_fmpc:
        sys.argv[1:] = ['--algo', 'fmpc_ext',
                        '--task', 'quadrotor',
                        '--overrides', yaml_file_base, yaml_file_fmpc]
        CONFIG_FACTORY = ConfigFactory()
        config = CONFIG_FACTORY.merge()

        env_func = partial(make, config.task, **config.task_config)
        ctrl_func = partial(make, config.algo, env_func, **config.algo_config)

        trajs, metrics, failed_runs = run_controller_trials('FMPC', env_func, ctrl_func, initial_states, gui)
        results_dict['fmpc'] = {'trajs_data': trajs, 'metrics': metrics, 'failed_runs': failed_runs}

    # Run FMPC+SOCP
    if run_fmpc_socp:
        sys.argv[1:] = ['--algo', 'fmpc_socp',
                        '--task', 'quadrotor',
                        '--overrides', yaml_file_base, yaml_file_fmpc_socp]
        CONFIG_FACTORY = ConfigFactory()
        config = CONFIG_FACTORY.merge()

        env_func = partial(make, config.task, **config.task_config)
        ctrl_func = partial(make, config.algo, env_func, **config.algo_config)

        trajs, metrics, failed_runs = run_controller_trials('FMPC+SOCP', env_func, ctrl_func, initial_states, gui)
        results_dict['fmpc_socp'] = {'trajs_data': trajs, 'metrics': metrics, 'failed_runs': failed_runs}

    # Save results
    save_results(output_dir, initial_states, seeds, results_dict)

    # Print summary
    print_summary_table(results_dict)

    # Generate plots
    plot_inference_time_violin(results_dict, output_dir)
    plot_tracking_error_distribution(results_dict, output_dir, ctrl_freq=50)
    plot_position_distribution(results_dict, output_dir,
                              is_constrained=(mode == 'constrained'),
                              constraint_state=-0.8)
    plot_input_distribution(results_dict, output_dir, ctrl_freq=50,
                           is_constrained=(mode == 'constrained'),
                           constraint_input=0.435)

    print(f'\n{"="*80}')
    print(f'Monte Carlo experiment completed!')
    print(f'Results saved to: {output_dir}')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Monte Carlo experiments with randomized initial conditions.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='normal',
        choices=['normal', 'constrained'],
        help='Choose the mode: normal or constrained (default: normal)'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=2,
        help='Number of Monte Carlo trials (default: 2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Show GUI during experiments'
    )
    parser.add_argument(
        '--controllers',
        type=str,
        nargs='+',
        default=['nmpc', 'fmpc', 'fmpc_socp'],
        choices=['nmpc', 'fmpc', 'fmpc_socp'],
        help='Which controllers to run (default: all)'
    )

    args = parser.parse_args()

    # Determine which controllers to run
    run_nmpc = 'nmpc' in args.controllers
    run_fmpc = 'fmpc' in args.controllers
    run_fmpc_socp = 'fmpc_socp' in args.controllers

    # Run the experiment
    run_monte_carlo_experiment(
        mode=args.mode,
        n_trials=args.n_trials,
        base_seed=args.seed,
        gui=args.gui,
        run_nmpc=run_nmpc,
        run_fmpc=run_fmpc,
        run_fmpc_socp=run_fmpc_socp
    )
