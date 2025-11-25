# Monte Carlo Experiments

This directory contains scripts for running Monte Carlo experiments with randomized initial conditions to obtain statistically significant performance comparisons between controllers.

## Overview

The Monte Carlo experiments run multiple trials (default: 2 for debugging, increase for production) with randomized initial conditions. Each controller (NMPC, FMPC, FMPC+SOCP) is tested on the **exact same** set of initial conditions to ensure fair comparison.

## Files

### Configuration Files

- `config_overrides_fast/` - Original fast (unconstrained) configurations
- `config_overrides_constrained/` - Original constrained configurations
- `config_overrides_fast_random/` - Fast configs with randomization enabled
- `config_overrides_constrained_random/` - Constrained configs with randomization enabled

The `*_random` directories contain minimal YAML overrides that enable `randomized_init: True` and define the randomization distributions.

### Scripts

- `run_exp_paper_monte_carlo.py` - Main Monte Carlo experiment script

## Usage

### Basic Usage

Run unconstrained experiments with 2 trials (debugging):
```bash
cd runs_paper/
python3 run_exp_paper_monte_carlo.py --mode normal --n_trials 2
```

Run constrained experiments with 2 trials:
```bash
python3 run_exp_paper_monte_carlo.py --mode constrained --n_trials 2
```

### Production Runs

For publication-quality statistics (50-100 trials):
```bash
# Unconstrained with 50 trials
python3 run_exp_paper_monte_carlo.py --mode normal --n_trials 50 --seed 42

# Constrained with 50 trials
python3 run_exp_paper_monte_carlo.py --mode constrained --n_trials 50 --seed 42
```

### Command-Line Options

- `--mode {normal,constrained}` - Experiment mode (default: normal)
- `--n_trials N` - Number of Monte Carlo trials (default: 2)
- `--seed SEED` - Base random seed for reproducibility (default: 42)
- `--gui` - Show GUI during experiments (not recommended for many trials)
- `--controllers {nmpc,fmpc,fmpc_socp} [...]` - Select which controllers to run (default: all)

### Examples

Run only FMPC+SOCP with 100 trials:
```bash
python3 run_exp_paper_monte_carlo.py --mode normal --n_trials 100 --controllers fmpc_socp
```

Run with custom seed:
```bash
python3 run_exp_paper_monte_carlo.py --mode normal --n_trials 30 --seed 12345
```

## Randomization Settings

Initial state randomization uses uniform distributions centered at nominal values:

| State Variable | Randomization Range |
|----------------|---------------------|
| `init_x`       | ±0.05 m            |
| `init_x_dot`   | ±0.05 m/s          |
| `init_z`       | ±0.05 m            |
| `init_z_dot`   | ±0.05 m/s          |
| `init_theta`   | ±0.05 rad (≈3°)    |
| `init_theta_dot` | ±0.05 rad/s      |

These values can be adjusted in the `config_overrides_*_random/quadrotor_2D_attitude_tracking.yaml` files.

## Output

Results are saved to `monte_carlo_results/{mode}/`:

```
monte_carlo_results/
├── normal/
│   ├── initial_states.pkl          # Generated initial states and seeds
│   ├── nmpc_trials.pkl              # NMPC trajectory data and metrics
│   ├── fmpc_trials.pkl              # FMPC trajectory data and metrics
│   ├── fmpc_socp_trials.pkl         # FMPC+SOCP trajectory data and metrics
│   ├── aggregated_metrics.pkl       # Summary statistics across all controllers
│   └── plots/                       # (for future visualization scripts)
└── constrained/
    └── (same structure)
```

### Metrics Computed

For each controller, the following metrics are aggregated across all trials:

- **Average RMSE** - Mean tracking error ± standard deviation
- **Average Inference Time** - Mean controller computation time
- **Failure Rate** - Percentage of trials with constraint violations
- **Constraint Violations** - Average number of violations per trial
- **Per-trial data** - Individual episode metrics for detailed analysis

### Reproducibility

All random number generation is seeded for reproducibility:
- Seeds are saved in `initial_states.pkl`
- Using the same `--seed` value will generate identical initial conditions
- Each trial uses seed = `base_seed + trial_number`

## Workflow

The script performs the following steps:

1. **Generate Initial States**: Create N randomized initial conditions using the `*_random` config files
2. **Run Controllers**: For each controller (NMPC, FMPC, FMPC+SOCP):
   - Reset environment with each initial state
   - Run one episode
   - Collect trajectory data
3. **Aggregate Metrics**: Compute mean, std dev, and other statistics across all trials
4. **Save Results**: Store all data and metrics to disk

## Notes

- **Computation Time**: Each trial takes ~2 minutes, so 50 trials × 3 controllers ≈ 5 hours
- **Fair Comparison**: All controllers see identical initial conditions for each trial
- **Statistical Significance**: 30+ trials recommended for CLT, 50-100 for publication
- **Debugging**: Start with `--n_trials 2` to verify everything works before long runs
- **Parallel Execution**: Currently sequential; could be parallelized in the future

## Example Output

```
================================================================================
Monte Carlo Experiment: NORMAL mode, 2 trials
================================================================================

Generating 2 randomized initial states...
  Trial 1/2: seed=42, state=[...]
  Trial 2/2: seed=43, state=[...]
Generated 2 initial states.

================================================================================
Running NMPC on 2 trials...
================================================================================
  Trial 1/2 - init_state: [...]
    Trial 1 completed.
  Trial 2/2 - init_state: [...]
    Trial 2 completed.

Computing aggregated metrics for NMPC...
NMPC completed: 2 trials
  Average RMSE: 0.0234 ± 0.0012
  Average inference time: 0.0123s
  Failure rate: 0.00%

[... FMPC and FMPC+SOCP results ...]

================================================================================
SUMMARY: Monte Carlo Experiment Results
================================================================================

Metric                         |            NMPC |            FMPC |      FMPC+SOCP
--------------------------------------------------------------------------------
Average RMSE (m)               |          0.0234 |          0.0245 |          0.0238
RMSE Std Dev (m)               |          0.0012 |          0.0015 |          0.0011
Avg Inference Time (ms)        |           12.34 |            8.56 |           15.67
Failure Rate (%)               |            0.00 |            0.00 |            0.00
Avg Constraint Violations      |            0.00 |            0.00 |            0.00
--------------------------------------------------------------------------------

Results saved to: ./monte_carlo_results/normal
```

## Future Enhancements

Potential additions:
- Automated plotting scripts for visualizing results with error bars
- Statistical significance tests (t-tests, ANOVA)
- Parallel execution for faster runs
- CSV export for easy analysis in Excel/MATLAB
- Confidence interval calculations
