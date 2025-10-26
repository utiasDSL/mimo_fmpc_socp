#!/bin/bash
# Convenient wrapper script for regenerating plots from Monte Carlo results
# Ensures the conda environment is activated before running

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install conda first."
    exit 1
fi

# Activate the safe conda environment
echo "Activating conda environment 'safe'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate safe

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'safe'"
    echo "Please create the environment first (see CLAUDE.md for installation instructions)"
    exit 1
fi

# Run the regenerate_plots.py script with all provided arguments
cd "$SCRIPT_DIR"
python3 regenerate_plots.py "$@"

# Deactivate conda environment
conda deactivate
