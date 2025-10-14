#!/usr/bin/env python3
"""Simple script to view Monte Carlo results."""

import pickle
import sys
import os

def main():
    if len(sys.argv) < 2:
        mode = 'normal'
    else:
        mode = sys.argv[1]

    results_dir = f'./monte_carlo_results/{mode}'

    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist!")
        return

    # Load aggregated metrics
    with open(os.path.join(results_dir, 'aggregated_metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)

    print(f"\n{'='*80}")
    print(f"Monte Carlo Results - {mode.upper()} mode")
    print(f"{'='*80}\n")

    for controller, controller_metrics in metrics.items():
        print(f"{controller.upper()}:")
        print(f"  Average RMSE: {controller_metrics.get('average_rmse', 0):.6f} ± {controller_metrics.get('rmse_std', 0):.6f}")
        print(f"  Failure rate: {controller_metrics.get('failure_rate', 0):.2%}")
        print()

if __name__ == '__main__':
    main()
