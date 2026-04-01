#!/bin/bash
# Regenerate plots with average RMSE appended to legend labels.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install conda first."
    exit 1
fi

echo "Activating conda environment 'safe'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate safe

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'safe'"
    exit 1
fi

cd "$SCRIPT_DIR"
python3 regenerate_plots.py --legend_rmse "$@"

conda deactivate
