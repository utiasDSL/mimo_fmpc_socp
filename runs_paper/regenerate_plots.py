#!/usr/bin/env python3
"""Regenerate plots from saved Monte Carlo experiment results.

This script loads previously saved Monte Carlo experiment data and regenerates
all plots using the same plotting functions as run_exp_paper_monte_carlo.py.

Usage:
    # Regenerate plots from a specific run
    python3 regenerate_plots.py --run_dir ./monte_carlo_results/normal/20250125_143022

    # Regenerate plots from the latest run in normal mode
    python3 regenerate_plots.py --mode normal --latest

    # Regenerate plots from the latest run in constrained mode
    python3 regenerate_plots.py --mode constrained --latest

    # Regenerate only specific plots
    python3 regenerate_plots.py --run_dir ./monte_carlo_results/normal/20250125_143022 --plots violin tracking

    # Save plots to a custom directory
    python3 regenerate_plots.py --run_dir ./monte_carlo_results/normal/20250125_143022 --output_dir ./custom_plots
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

# Import plotting functions from the standalone plotting module
from monte_carlo_plotting import (
    plot_inference_time_violin,
    plot_tracking_error_distribution,
    plot_position_distribution,
    plot_input_distribution,
    print_summary_table,
    print_timing_breakdown_table
)


def find_latest_run(mode='normal'):
    """Find the latest timestamped run directory for a given mode.

    Args:
        mode (str): 'normal' or 'constrained'

    Returns:
        str: Path to the latest run directory, or None if not found
    """
    base_dir = Path('./monte_carlo_results') / mode

    if not base_dir.exists():
        print(f'Error: Directory {base_dir} does not exist')
        return None

    # Get all timestamped directories (format: YYYYMMDD_HHMMSS)
    run_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.replace('_', '').isdigit()]

    if not run_dirs:
        print(f'Error: No run directories found in {base_dir}')
        return None

    # Sort by directory name (timestamp) and get the latest
    latest_dir = sorted(run_dirs, key=lambda d: d.name)[-1]

    print(f'Found latest run: {latest_dir}')
    return str(latest_dir)


def load_monte_carlo_results(run_dir):
    """Load Monte Carlo experiment results from a run directory.

    Args:
        run_dir (str): Path to the run directory containing pkl files

    Returns:
        dict: results_dict with structure:
            {
                'nmpc': {'trajs_data': ..., 'metrics': ..., 'failed_runs': ...},
                'fmpc': {'trajs_data': ..., 'metrics': ..., 'failed_runs': ...},
                'fmpc_socp': {'trajs_data': ..., 'metrics': ..., 'failed_runs': ...}
            }
        dict: initial_states_data with 'initial_states' and 'seeds'
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise ValueError(f'Run directory does not exist: {run_dir}')

    print(f'\n{"="*80}')
    print(f'Loading data from: {run_dir}')
    print(f'{"="*80}\n')

    # Load initial states
    initial_states_file = run_dir / 'initial_states.pkl'
    if not initial_states_file.exists():
        print(f'Warning: {initial_states_file} not found')
        initial_states_data = None
    else:
        with open(initial_states_file, 'rb') as f:
            initial_states_data = pickle.load(f)
        print(f'Loaded initial states: {len(initial_states_data["initial_states"])} trials')

    # Load controller results
    results_dict = {}
    controller_names = ['nmpc', 'fmpc', 'fmpc_socp']

    for controller_name in controller_names:
        trials_file = run_dir / f'{controller_name}_trials.pkl'

        if trials_file.exists():
            with open(trials_file, 'rb') as f:
                data = pickle.load(f)

            results_dict[controller_name] = data

            # Print loaded data info
            n_successful = data['metrics'].get('n_successful', 0)
            n_failed = data['metrics'].get('n_failed', 0)
            print(f'Loaded {controller_name.upper()}: {n_successful} successful, {n_failed} failed trials')
        else:
            print(f'Skipping {controller_name.upper()}: {trials_file} not found')

    if not results_dict:
        raise ValueError(f'No controller data found in {run_dir}')

    print(f'\nLoaded {len(results_dict)} controller(s): {list(results_dict.keys())}')
    print()

    return results_dict, initial_states_data


def detect_experiment_mode(run_dir):
    """Detect if the experiment was run in 'normal' or 'constrained' mode.

    Args:
        run_dir (str): Path to the run directory

    Returns:
        str: 'normal' or 'constrained'
    """
    run_dir = Path(run_dir)

    # Check if run_dir contains 'normal' or 'constrained' in path
    if 'constrained' in str(run_dir):
        return 'constrained'
    elif 'normal' in str(run_dir):
        return 'normal'

    # Check README.txt if it exists
    readme_file = run_dir / 'README.txt'
    if readme_file.exists():
        with open(readme_file, 'r') as f:
            content = f.read()
            if 'Mode: constrained' in content:
                return 'constrained'
            elif 'Mode: normal' in content:
                return 'normal'

    # Default to normal
    print('Warning: Could not detect experiment mode, assuming "normal"')
    return 'normal'


def regenerate_plots(run_dir, output_dir=None, plot_types=None, ctrl_freq=50):
    """Regenerate all plots from saved Monte Carlo results.

    Args:
        run_dir (str): Path to the run directory containing saved results
        output_dir (str): Directory to save plots (defaults to run_dir)
        plot_types (list): List of plot types to generate. Options:
            - 'violin': Inference time violin plot
            - 'tracking': Tracking error distribution plot
            - 'position': Position distribution plot
            - 'input': Input distribution plot
            - 'summary': Print summary tables (not saved as image)
            If None, generate all plots.
        ctrl_freq (int): Control frequency in Hz (for time axis)
    """
    # Load data
    results_dict, initial_states_data = load_monte_carlo_results(run_dir)

    # Detect experiment mode (normal or constrained)
    mode = detect_experiment_mode(run_dir)
    is_constrained = (mode == 'constrained')

    print(f'Detected experiment mode: {mode}')

    # Set output directory
    if output_dir is None:
        output_dir = run_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Saving plots to: {output_dir}\n')

    # Determine which plots to generate
    all_plot_types = ['violin', 'tracking', 'position', 'input', 'summary']
    if plot_types is None:
        plot_types = all_plot_types
    else:
        # Validate plot types
        invalid = set(plot_types) - set(all_plot_types)
        if invalid:
            raise ValueError(f'Invalid plot types: {invalid}. Valid options: {all_plot_types}')

    print(f'{"="*80}')
    print(f'Generating plots: {plot_types}')
    print(f'{"="*80}\n')

    # Generate plots
    if 'summary' in plot_types:
        print('Generating summary tables...')
        print_summary_table(results_dict)

    if 'violin' in plot_types:
        print('Generating inference time violin plot...')
        plot_inference_time_violin(results_dict, str(output_dir))

    if 'tracking' in plot_types:
        print('Generating tracking error distribution plot...')
        plot_tracking_error_distribution(results_dict, str(output_dir), ctrl_freq=ctrl_freq)

    if 'position' in plot_types:
        print('Generating position distribution plot...')
        plot_position_distribution(
            results_dict,
            str(output_dir),
            is_constrained=is_constrained,
            constraint_state=-0.8
        )

    if 'input' in plot_types:
        print('Generating input distribution plot...')
        plot_input_distribution(
            results_dict,
            str(output_dir),
            ctrl_freq=ctrl_freq,
            is_constrained=is_constrained,
            constraint_input=0.435
        )

    print(f'\n{"="*80}')
    print(f'Plots regenerated successfully!')
    print(f'Output directory: {output_dir}')
    print(f'{"="*80}\n')


def list_available_runs(mode=None):
    """List all available Monte Carlo run directories.

    Args:
        mode (str): If specified, only list runs for this mode ('normal' or 'constrained')
    """
    base_dir = Path('./monte_carlo_results')

    if not base_dir.exists():
        print(f'No Monte Carlo results found in {base_dir}')
        return

    modes = [mode] if mode else ['normal', 'constrained']

    print(f'\n{"="*80}')
    print('Available Monte Carlo Runs')
    print(f'{"="*80}\n')

    for m in modes:
        mode_dir = base_dir / m
        if not mode_dir.exists():
            continue

        run_dirs = sorted([d for d in mode_dir.iterdir() if d.is_dir()],
                         key=lambda d: d.name, reverse=True)

        if run_dirs:
            print(f'{m.upper()} mode runs:')
            for run_dir in run_dirs:
                # Try to get run info from README
                readme_file = run_dir / 'README.txt'
                info = ''
                if readme_file.exists():
                    with open(readme_file, 'r') as f:
                        for line in f:
                            if 'Timestamp:' in line:
                                info = line.split('Timestamp:', 1)[1].strip()
                                break

                print(f'  {run_dir.relative_to(base_dir)}  {info}')
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Regenerate plots from saved Monte Carlo experiment results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate from specific run directory
  python3 regenerate_plots.py --run_dir ./monte_carlo_results/normal/20250125_143022

  # Regenerate from latest normal run
  python3 regenerate_plots.py --mode normal --latest

  # Regenerate only violin and tracking plots
  python3 regenerate_plots.py --run_dir ./monte_carlo_results/normal/20250125_143022 --plots violin tracking

  # List all available runs
  python3 regenerate_plots.py --list
        """
    )

    parser.add_argument(
        '--run_dir',
        type=str,
        help='Path to the run directory containing saved results'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['normal', 'constrained'],
        help='Experiment mode (used with --latest to find latest run)'
    )

    parser.add_argument(
        '--latest',
        action='store_true',
        help='Use the latest run directory for the specified mode'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='Directory to save regenerated plots (default: same as run_dir)'
    )

    parser.add_argument(
        '--plots',
        type=str,
        nargs='+',
        choices=['violin', 'tracking', 'position', 'input', 'summary'],
        help='Which plots to generate (default: all). Options: violin, tracking, position, input, summary'
    )

    parser.add_argument(
        '--ctrl_freq',
        type=int,
        default=50,
        help='Control frequency in Hz for time axis (default: 50)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available Monte Carlo run directories'
    )

    args = parser.parse_args()

    # Handle --list option
    if args.list:
        list_available_runs()
        sys.exit(0)

    # Determine run directory
    run_dir = None

    if args.latest:
        if not args.mode:
            print('Error: --mode must be specified when using --latest')
            sys.exit(1)
        run_dir = find_latest_run(args.mode)
        if run_dir is None:
            sys.exit(1)
    elif args.run_dir:
        run_dir = args.run_dir
    else:
        print('Error: Either --run_dir or --latest must be specified')
        print('Use --list to see available runs')
        sys.exit(1)

    # Regenerate plots
    try:
        regenerate_plots(
            run_dir=run_dir,
            output_dir=args.output_dir,
            plot_types=args.plots,
            ctrl_freq=args.ctrl_freq
        )
    except Exception as e:
        print(f'\nError: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
