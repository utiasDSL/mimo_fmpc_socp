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
from datetime import datetime
import shutil

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

# Import plotting and analysis functions from centralized module
from monte_carlo_plotting import (
    compute_second_half_rmse,
    extract_timing_statistics,
    print_timing_breakdown_table,
    print_summary_table,
    plot_inference_time_violin,
    plot_tracking_error_distribution,
    plot_position_distribution,
    plot_input_distribution
)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


class TeeLogger:
    """Captures stdout/stderr and writes to both console and log file."""

    def __init__(self, log_filepath, stream):
        """Initialize the tee logger.

        Args:
            log_filepath (str): Path to log file
            stream: Original stream (sys.stdout or sys.stderr)
        """
        self.log_file = open(log_filepath, 'a')
        self.stream = stream

    def write(self, message):
        """Write message to both console and file."""
        self.stream.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        """Flush both streams."""
        self.stream.flush()
        self.log_file.flush()

    def close(self):
        """Close the log file."""
        self.log_file.close()


def setup_logging(run_dir):
    """Set up logging to capture all terminal output.

    Args:
        run_dir (str): Directory to save log file

    Returns:
        tuple: (stdout_logger, stderr_logger) - Loggers for cleanup
    """
    log_filepath = os.path.join(run_dir, 'run.log')

    # Write header to log file
    with open(log_filepath, 'w') as f:
        f.write('='*80 + '\n')
        f.write('Monte Carlo Experiment Run Log\n')
        f.write('='*80 + '\n')
        f.write(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('='*80 + '\n\n')

    # Redirect stdout and stderr to log file while keeping console output
    stdout_logger = TeeLogger(log_filepath, sys.stdout)
    stderr_logger = TeeLogger(log_filepath, sys.stderr)

    sys.stdout = stdout_logger
    sys.stderr = stderr_logger

    print(f'Logging terminal output to: {log_filepath}\n')

    return stdout_logger, stderr_logger


def cleanup_logging(stdout_logger, stderr_logger, original_stdout, original_stderr):
    """Restore original stdout/stderr and close log files.

    Args:
        stdout_logger: The TeeLogger for stdout
        stderr_logger: The TeeLogger for stderr
        original_stdout: Original sys.stdout
        original_stderr: Original sys.stderr
    """
    # Write footer to log
    print(f'\n{"="*80}')
    print(f'Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*80}')

    # Restore original streams
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Close log files
    stdout_logger.close()
    stderr_logger.close()

    print(f'\nLog file saved successfully.')


def create_timestamped_run_directory(base_dir, mode):
    """Create a timestamped directory for this Monte Carlo run.

    Args:
        base_dir (str): Base directory (e.g., './monte_carlo_results')
        mode (str): Experiment mode ('normal' or 'constrained')

    Returns:
        str: Path to the timestamped run directory
    """
    # Create base directory if it doesn't exist
    mode_dir = os.path.join(base_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)

    # Create timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create timestamped run directory
    run_dir = os.path.join(mode_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    print(f'\n{"="*80}')
    print(f'Created run directory: {run_dir}')
    print(f'{"="*80}\n')

    return run_dir


def copy_config_files(run_dir, config_files, mode):
    """Copy all config files used in this run to the run directory.

    Args:
        run_dir (str): Timestamped run directory
        config_files (list): List of config file paths used in the run
        mode (str): Experiment mode (for organizing configs)
    """
    # Create configs subdirectory
    configs_dir = os.path.join(run_dir, 'configs')
    os.makedirs(configs_dir, exist_ok=True)

    print(f'\nCopying config files to {configs_dir}:')

    for config_file in config_files:
        if os.path.exists(config_file):
            # Get just the filename
            filename = os.path.basename(config_file)

            # Copy to configs directory
            dest_path = os.path.join(configs_dir, filename)
            shutil.copy2(config_file, dest_path)
            print(f'  ✓ {filename}')
        else:
            print(f'  ✗ Warning: {config_file} not found')

    # Create a README with run information
    readme_path = os.path.join(run_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write(f'Monte Carlo Experiment Run\n')
        f.write(f'{"="*80}\n\n')
        f.write(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Mode: {mode}\n')
        f.write(f'\nConfig files used:\n')
        for config_file in config_files:
            f.write(f'  - {os.path.basename(config_file)}\n')
        f.write(f'\nDirectory structure:\n')
        f.write(f'  ./run.log            - Complete terminal output log\n')
        f.write(f'  ./configs/           - Configuration files\n')
        f.write(f'  ./*.pkl             - Pickled data (trajectories, metrics, states)\n')
        f.write(f'  ./*.pdf, ./*.png    - Plots and figures\n')

    print(f'  ✓ README.txt\n')


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

    # Get expected episode length
    temp_env = env_func(gui=False)
    MAX_STEPS = int(temp_env.CTRL_FREQ * temp_env.EPISODE_LEN_SEC)
    temp_env.close()
    print(f'Expected episode length: {MAX_STEPS} steps\n')

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

            # Load pre-trained GP models for GPMPC if available
            if hasattr(ctrl, 'load') and hasattr(ctrl, 'gp_model_path') and ctrl.gp_model_path:
                print(f'    Loading pre-trained GP models from {ctrl.gp_model_path}')
                ctrl.load(ctrl.gp_model_path, do_reset=False)
                experiment = BaseExperiment(env=env, ctrl=ctrl, train_env=train_env, initial_ctrl_reset=False)
            else:
                experiment = BaseExperiment(env=env, ctrl=ctrl, train_env=train_env, initial_ctrl_reset=True)

            # Run evaluation for this trial (1 episode)
            trajs_data, _ = experiment.run_evaluation(training=False, n_episodes=1, verbose=False, initial_reset=False)

            # Check if episode completed full duration (detect early termination)
            actual_steps = len(trajs_data['obs'][0]) if len(trajs_data['obs']) > 0 else 0

            if actual_steps < MAX_STEPS:
                # Early termination detected - treat as failure
                raise RuntimeError(
                    f'Episode terminated early at step {actual_steps}/{MAX_STEPS}. '
                    f'This may indicate constraint violations, out-of-bounds states, or other failures.'
                )

            # Success: collect trajectory data
            for key, value in trajs_data.items():
                successful_trajs[key] += value

            print(f'    Trial {trial_idx+1} completed successfully ({actual_steps} steps).\n')

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


# Note: print_timing_breakdown_table is now imported from monte_carlo_plotting


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


# Note: All plotting and analysis functions (print_summary_table, plot_*) are now
# imported from monte_carlo_plotting to ensure consistency between experiment runs
# and plot regeneration.

def run_monte_carlo_experiment(mode='normal', n_trials=2, base_seed=42, gui=False,
                               run_nmpc=True, run_fmpc=True, run_fmpc_socp=True, run_gpmpc=True):
    """Main function to run Monte Carlo experiments.

    Args:
        mode (str): 'normal' or 'constrained'
        n_trials (int): Number of Monte Carlo trials
        base_seed (int): Base seed for reproducibility
        gui (bool): Whether to show GUI
        run_nmpc (bool): Whether to run NMPC controller
        run_fmpc (bool): Whether to run FMPC controller
        run_fmpc_socp (bool): Whether to run FMPC+SOCP controller
        run_gpmpc (bool): Whether to run GPMPC controller
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
        yaml_file_gpmpc = './config_overrides_fast/gpmpc_acados_TP_quadrotor_2D_attitude_tracking.yaml'
    elif mode == 'constrained':
        yaml_file_base = './config_overrides_constrained/quadrotor_2D_attitude_tracking.yaml'
        yaml_file_base_random = './config_overrides_constrained_random/quadrotor_2D_attitude_tracking.yaml'
        yaml_file_nmpc = './config_overrides_constrained/mpc_quadrotor_2D_attitude_tracking.yaml'
        yaml_file_fmpc = './config_overrides_constrained/fmpc_ext_quadrotor_2D_attitude_tracking.yaml'
        yaml_file_fmpc_socp = './config_overrides_constrained/fmpc_socp_quadrotor_2D_attitude_tracking.yaml'
        yaml_file_gpmpc = './config_overrides_constrained/gpmpc_acados_TP_quadrotor_2D_attitude_tracking.yaml'
    else:
        raise ValueError(f'Unknown mode: {mode}')

    # Collect all config files
    config_files = [yaml_file_base, yaml_file_base_random, yaml_file_nmpc,
                    yaml_file_fmpc, yaml_file_fmpc_socp, yaml_file_gpmpc]

    # Verify config files exist
    for yaml_file in config_files:
        assert os.path.exists(yaml_file), f'{yaml_file} does not exist'

    # Create timestamped run directory
    output_dir = create_timestamped_run_directory('./monte_carlo_results', mode)

    # Copy config files to run directory
    copy_config_files(output_dir, config_files, mode)

    # Set up logging to capture all terminal output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_logger, stderr_logger = setup_logging(output_dir)

    try:
        # Main experiment execution wrapped in try-finally for cleanup
        run_monte_carlo_experiment_impl(mode, n_trials, base_seed, gui, run_nmpc, run_fmpc,
                                        run_fmpc_socp, run_gpmpc, output_dir, yaml_file_base,
                                        yaml_file_base_random, yaml_file_nmpc, yaml_file_fmpc,
                                        yaml_file_fmpc_socp, yaml_file_gpmpc)
    finally:
        # Always restore logging even if experiment fails
        cleanup_logging(stdout_logger, stderr_logger, original_stdout, original_stderr)


def run_monte_carlo_experiment_impl(mode, n_trials, base_seed, gui, run_nmpc, run_fmpc,
                                     run_fmpc_socp, run_gpmpc, output_dir, yaml_file_base,
                                     yaml_file_base_random, yaml_file_nmpc, yaml_file_fmpc,
                                     yaml_file_fmpc_socp, yaml_file_gpmpc):
    """Implementation of Monte Carlo experiment (separated for logging wrapper).

    This function contains the actual experiment logic and is called by
    run_monte_carlo_experiment after logging is set up.
    """
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

    # Run GPMPC
    if run_gpmpc:
        sys.argv[1:] = ['--algo', 'gpmpc_acados_TP',
                        '--task', 'quadrotor',
                        '--overrides', yaml_file_base, yaml_file_gpmpc]
        CONFIG_FACTORY = ConfigFactory()
        config = CONFIG_FACTORY.merge()

        env_func = partial(make, config.task, **config.task_config)
        ctrl_func = partial(make, config.algo, env_func, **config.algo_config)

        trajs, metrics, failed_runs = run_controller_trials('GPMPC', env_func, ctrl_func, initial_states, gui)
        results_dict['gpmpc'] = {'trajs_data': trajs, 'metrics': metrics, 'failed_runs': failed_runs}

    # Save results
    save_results(output_dir, initial_states, seeds, results_dict)

    # Print summary
    print_summary_table(results_dict)

    # Generate plots
    plot_inference_time_violin(results_dict, output_dir,
                              is_constrained=(mode == 'constrained'))
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
        default=['nmpc', 'fmpc', 'fmpc_socp', 'gpmpc'],
        choices=['nmpc', 'fmpc', 'fmpc_socp', 'gpmpc'],
        help='Which controllers to run (default: all)'
    )

    args = parser.parse_args()

    # Determine which controllers to run
    run_nmpc = 'nmpc' in args.controllers
    run_fmpc = 'fmpc' in args.controllers
    run_fmpc_socp = 'fmpc_socp' in args.controllers
    run_gpmpc = 'gpmpc' in args.controllers

    # Run the experiment
    run_monte_carlo_experiment(
        mode=args.mode,
        n_trials=args.n_trials,
        base_seed=args.seed,
        gui=args.gui,
        run_nmpc=run_nmpc,
        run_fmpc=run_fmpc,
        run_fmpc_socp=run_fmpc_socp,
        run_gpmpc=run_gpmpc
    )
