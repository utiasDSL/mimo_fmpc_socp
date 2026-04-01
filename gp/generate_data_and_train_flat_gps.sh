#!/bin/bash
set -euo pipefail

# Run from the script directory so all relative paths inside the Python scripts
# resolve against gp/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Generate the simulation-based figure-8 train/test datasets.
python3 fgp_training_data_sim_ext.py

# 2. Densify the figure-8 training set with noisy samples around the trajectory.
python3 fgp_training_data_aroundFig8.py

# 3. Train the first flat GP used by FMPC+SOCP (v0). This model uses the
# longer 5000-iteration budget and a higher initial noise to bias training
# toward better uncertainty calibration.
python3 training_flat_gps.py --gp 0 --iterations 5000 --initial_noise 8

# 4. Train the second flat GP used by FMPC+SOCP (v1). Keep this one at the
# original 2000-iteration budget to match the intended default artifact.
python3 training_flat_gps.py --gp 1 --iterations 2000
