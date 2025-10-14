# runs FMPC+SOCP experiment with line profiling
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for profiling
from mpc_experiment_paper import run

parser = argparse.ArgumentParser(description="Run FMPC+SOCP with profiling.")
parser.add_argument(
    '--mode',
    type=str,
    default='normal',
    choices=['normal', 'constrained'],
    help='Choose the mode: normal or constrained (default is normal).'
)

args = parser.parse_args()

if args.mode == 'normal':
    print("Running FMPC+SOCP with unconstrained lemniscate (profiling mode).")
    yaml_file_base = './config_overrides_fast/quadrotor_2D_attitude_tracking.yaml'
    yaml_file_fmpc_socp = './config_overrides_fast/fmpc_socp_quadrotor_2D_attitude_tracking.yaml'
elif args.mode == 'constrained':
    print("Running FMPC+SOCP with constrained lemniscate (profiling mode).")
    yaml_file_base = './config_overrides_constrained/quadrotor_2D_attitude_tracking.yaml'
    yaml_file_fmpc_socp = './config_overrides_constrained/fmpc_socp_quadrotor_2D_attitude_tracking.yaml'

GUI = False
data_path_fmpc_socp = './temp-data/fmpc_socp_data_quadrotor_traj_tracking.pkl'

# Run FMPC+SOCP
if os.path.exists(data_path_fmpc_socp):
    os.remove(data_path_fmpc_socp)
    print(f"{data_path_fmpc_socp} deleted successfully.")
else:
    print(f"{data_path_fmpc_socp} does not exist.")

run(gui=GUI, save_data=True, algo='fmpc_socp', yaml_base=yaml_file_base, yaml_ctrl=yaml_file_fmpc_socp)

print('\nProfiling complete! Check the output above for line-by-line timing.')
