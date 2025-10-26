#!/bin/bash
# Launcher script for Monte Carlo Data Viewer
# Ensures the correct conda environment is activated before launching the GUI

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda."
    exit 1
fi

# Check if 'safe' environment exists
if ! conda env list | grep -q "^safe "; then
    echo "Error: conda environment 'safe' not found."
    echo "Please create it using:"
    echo "  conda create -n safe python=3.10"
    echo "  conda activate safe"
    echo "  cd $(dirname $SCRIPT_DIR)"
    echo "  pip install -e ."
    exit 1
fi

# Activate the safe environment and launch the viewer
echo "Activating conda environment 'safe'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate safe

echo "Launching Monte Carlo Data Viewer..."
cd "$SCRIPT_DIR"
python monte_carlo_viewer.py

# Keep conda environment active
exec bash
