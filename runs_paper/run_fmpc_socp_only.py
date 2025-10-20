# runs FMPC+SOCP experiment only
import os
import argparse
from mpc_experiment_paper import run
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

parser = argparse.ArgumentParser(description="Run FMPC+SOCP only.")
parser.add_argument(
    '--mode',
    type=str,
    default='normal',
    choices=['normal', 'constrained'],
    help='Choose the mode: normal or constrained (default is normal).'
)

args = parser.parse_args()

if args.mode == 'normal':
    print("Running FMPC+SOCP with unconstrained lemniscate.")
    yaml_file_base = './config_overrides_fast/quadrotor_2D_attitude_tracking.yaml'
    yaml_file_fmpc_socp = './config_overrides_fast/fmpc_socp_quadrotor_2D_attitude_tracking.yaml'
elif args.mode == 'constrained':
    print("Running FMPC+SOCP with constrained lemniscate.")
    yaml_file_base = './config_overrides_constrained/quadrotor_2D_attitude_tracking.yaml'
    yaml_file_fmpc_socp = './config_overrides_constrained/fmpc_socp_quadrotor_2D_attitude_tracking.yaml'

GUI = False
ctrl_freq = 50
sample_time = 1/ctrl_freq
num_loops = 2

data_path_fmpc_socp = './temp-data/fmpc_socp_data_quadrotor_traj_tracking.pkl'

# Run FMPC+SOCP
if os.path.exists(data_path_fmpc_socp):
    os.remove(data_path_fmpc_socp)
    print(f"{data_path_fmpc_socp} deleted successfully.")
else:
    print(f"{data_path_fmpc_socp} does not exist.")

run(gui=GUI, save_data=True, algo='fmpc_socp', yaml_base=yaml_file_base, yaml_ctrl=yaml_file_fmpc_socp)

# Extract data
def extract_data(data_file):
    with open(data_file, 'rb') as file:
        data_dict = pickle.load(file)
    metrics = data_dict['metrics']
    traj_data = data_dict['trajs_data']
    states = traj_data['obs'][0]

    mse_dict = []
    for info in traj_data['info'][0]:
        if 'mse' in info:
            mse_dict.append(info.get('mse'))
        else:
            mse_dict.append(0)
    error = np.array(mse_dict)
    rmse = metrics['rmse']

    inference_time = np.array(traj_data['inference_time_data'])
    state_ref = np.array(traj_data['controller_data'][0]['goal_states'])
    action = traj_data['action'][0]

    ctrl_data = traj_data['controller_data'][0]
    if 'u_extSOCP' in ctrl_data:
        action_ext = 0
        gp_time = np.array(ctrl_data.get('gp_time', [[0]])[0])
        socp_solve_time = np.array(ctrl_data.get('socp_solve_time', [[0]])[0])
        mpc_solve_time = np.array(ctrl_data.get('mpc_solve_time', [0]))
        observer_time = np.array(ctrl_data.get('observer_time', [0]))
        flat_transform_time = np.array(ctrl_data.get('flat_transform_time', [0]))
        cholesky_time = np.array(ctrl_data.get('cholesky_time', [[0]])[0])
        cost_time = np.array(ctrl_data.get('cost_time', [0]))
        dummy_time = np.array(ctrl_data.get('dummy_time', [0]))
        stab_time = np.array(ctrl_data.get('stab_time', [0]))
        state_time = np.array(ctrl_data.get('state_time', [0]))
        param_time = np.array(ctrl_data.get('param_time', [0]))
        dyn_ext_time = np.array(ctrl_data.get('dyn_ext_time', [0]))
        observer_update_time = np.array(ctrl_data.get('observer_update_time', [0]))
        logging_time = np.array(ctrl_data.get('logging_time', [0]))
        # New timing metrics
        total_select_action_time = np.array(ctrl_data.get('total_select_action_time', [0]))
        mpc_total_time = np.array(ctrl_data.get('mpc_total_time', [0]))
        socp_total_time = np.array(ctrl_data.get('socp_total_time', [0]))
        overhead_time = np.array(ctrl_data.get('overhead_time', [0]))
        # MPC warm-start timing breakdown
        mpc_extract_time_1 = np.array(ctrl_data.get('mpc_extract_time_1', [0]))
        mpc_extract_time_2 = np.array(ctrl_data.get('mpc_extract_time_2', [0]))
        mpc_logging_time_1 = np.array(ctrl_data.get('mpc_logging_time_1', [0]))
        mpc_logging_time_2 = np.array(ctrl_data.get('mpc_logging_time_2', [0]))
        mpc_state_extract_time = np.array(ctrl_data.get('mpc_state_extract_time', [0]))
        mpc_input_extract_time = np.array(ctrl_data.get('mpc_input_extract_time', [0]))
        mpc_deepcopy_state_time = np.array(ctrl_data.get('mpc_deepcopy_state_time', [0]))
        mpc_deepcopy_input_time = np.array(ctrl_data.get('mpc_deepcopy_input_time', [0]))
        mpc_deepcopy_goal_time = np.array(ctrl_data.get('mpc_deepcopy_goal_time', [0]))
    else:
        action_ext = 0
        gp_time = 0
        socp_solve_time = 0
        mpc_solve_time = 0
        observer_time = 0
        flat_transform_time = 0
        cholesky_time = 0
        cost_time = 0
        dummy_time = 0
        stab_time = 0
        state_time = 0
        param_time = 0
        dyn_ext_time = 0
        observer_update_time = 0
        logging_time = 0
        total_select_action_time = 0
        mpc_total_time = 0
        socp_total_time = 0
        overhead_time = 0
        mpc_extract_time_1 = 0
        mpc_extract_time_2 = 0
        mpc_logging_time_1 = 0
        mpc_logging_time_2 = 0
        mpc_state_extract_time = 0
        mpc_input_extract_time = 0
        mpc_deepcopy_state_time = 0
        mpc_deepcopy_input_time = 0
        mpc_deepcopy_goal_time = 0

    return (states, error, inference_time, rmse, state_ref, action, action_ext, gp_time, socp_solve_time,
            mpc_solve_time, observer_time, flat_transform_time, cholesky_time, cost_time, dummy_time,
            stab_time, state_time, param_time, dyn_ext_time, observer_update_time, logging_time,
            total_select_action_time, mpc_total_time, socp_total_time, overhead_time,
            mpc_extract_time_1, mpc_extract_time_2, mpc_logging_time_1, mpc_logging_time_2,
            mpc_state_extract_time, mpc_input_extract_time, mpc_deepcopy_state_time,
            mpc_deepcopy_input_time, mpc_deepcopy_goal_time)

(state_fmpc_socp, error_fmpc_socp, inf_time_fmpc_socp, rmse_fmpc_socp, state_ref_fmpc_socp, action_fmpc_socp, _,
 gp_time_socp, socp_solve_time, mpc_solve_time, observer_time, flat_transform_time, cholesky_time,
 cost_time, dummy_time, stab_time, state_time, param_time, dyn_ext_time, observer_update_time,
 logging_time, total_select_action_time, mpc_total_time, socp_total_time, overhead_time,
 mpc_extract_time_1, mpc_extract_time_2, mpc_logging_time_1, mpc_logging_time_2,
 mpc_state_extract_time, mpc_input_extract_time, mpc_deepcopy_state_time,
 mpc_deepcopy_input_time, mpc_deepcopy_goal_time) = extract_data(data_path_fmpc_socp)

# Print metrics
end_idx_first_loop = int(np.shape(state_fmpc_socp)[0]/num_loops)

def compute_tracking_error(error, end_idx_first_loop):
    mean_track_err = np.mean(np.sqrt(error))
    loop1_track_err = np.mean(np.sqrt(error[:end_idx_first_loop]))
    loop2_track_err = np.mean(np.sqrt(error[end_idx_first_loop:]))
    return mean_track_err, loop1_track_err, loop2_track_err

mean_track_err, loop1_track_err, loop2_track_err = compute_tracking_error(error_fmpc_socp, end_idx_first_loop)

print('\n=== FMPC+SOCP Results ===')
print('\nTracking Error: mean(sqrt(sum of squares at each timestep))')
print(' average track_err: {:.2f}mm'.format(mean_track_err*1000))
print('1st loop track_err: {:.2f}mm'.format(loop1_track_err*1000))
print('2nd loop track_err: {:.2f}mm'.format(loop2_track_err*1000))

print('\nRMSE: sqrt(mean(sum of squares at each timestep))')
print('      average RMSE: {:.2f}mm'.format(rmse_fmpc_socp*1000))

print('\n' + '='*60)
print('TIMING BREAKDOWN')
print('='*60)

print('\n--- High-Level Timing ---')
print('Total inference time:     {:.2f}ms (avg)  {:.2f}ms (max)'.format(
    np.mean(inf_time_fmpc_socp)*1000, np.max(inf_time_fmpc_socp)*1000))
print('MPC solve time:           {:.2f}ms (avg)  {:.2f}ms (min)  {:.2f}ms (max)'.format(
    np.mean(mpc_solve_time)*1000, np.min(mpc_solve_time)*1000, np.max(mpc_solve_time)*1000))
print('Observer (compute):       {:.2f}ms (avg)'.format(np.mean(observer_time)*1000))
print('Observer (update):        {:.2f}ms (avg)'.format(np.mean(observer_update_time)*1000))
print('Flat transformation:      {:.2f}ms (avg)'.format(np.mean(flat_transform_time)*1000))
print('Dynamic extension:        {:.2f}ms (avg)'.format(np.mean(dyn_ext_time)*1000))
print('Data logging:             {:.2f}ms (avg)'.format(np.mean(logging_time)*1000))

print('\n--- SOCP Filter Breakdown ---')
print('GP inference (pure):      {:.2f}ms (avg)'.format(np.mean(gp_time_socp)*1000))
print('Cholesky decomp + inv:    {:.2f}ms (avg)'.format(np.mean(cholesky_time)*1000))
print('Cost computation:         {:.2f}ms (avg)'.format(np.mean(cost_time)*1000))
print('Dummy matrices:           {:.2f}ms (avg)'.format(np.mean(dummy_time)*1000))
print('Stability matrices:       {:.2f}ms (avg)'.format(np.mean(stab_time)*1000))
print('State constraint matrices:{:.2f}ms (avg)'.format(np.mean(state_time)*1000))
print('Parameter assignment:     {:.2f}ms (avg)'.format(np.mean(param_time)*1000))
print('SOCP solve:               {:.2f}ms (avg)  {:.2f}ms (min)  {:.2f}ms (max)'.format(
    np.mean(socp_solve_time)*1000, np.min(socp_solve_time)*1000, np.max(socp_solve_time)*1000))

print('\n--- Computed Sums ---')
socp_setup_time = (np.mean(gp_time_socp) + np.mean(cholesky_time) + np.mean(cost_time) +
                   np.mean(dummy_time) + np.mean(stab_time) + np.mean(state_time) + np.mean(param_time))
measured_components = (np.mean(mpc_solve_time) + np.mean(observer_time) + np.mean(observer_update_time) +
                      np.mean(flat_transform_time) + np.mean(dyn_ext_time) + np.mean(logging_time) +
                      socp_setup_time + np.mean(socp_solve_time))
overhead = np.mean(inf_time_fmpc_socp) - measured_components

print('SOCP setup (all):         {:.2f}ms'.format(socp_setup_time*1000))
print('Sum of measured parts:    {:.2f}ms'.format(measured_components*1000))
print('Unaccounted overhead:     {:.2f}ms ({:.1f}%)'.format(
    overhead*1000, 100*overhead/np.mean(inf_time_fmpc_socp)))

print('\n--- Overhead Analysis (from select_action) ---')
print('Total select_action time: {:.2f}ms (avg)'.format(np.mean(total_select_action_time)*1000))
print('MPC total (incl overhead):{:.2f}ms (avg)'.format(np.mean(mpc_total_time)*1000))
print('SOCP total (incl overhead):{:.2f}ms (avg)'.format(np.mean(socp_total_time)*1000))
print('Python/function overhead: {:.2f}ms (avg)  {:.2f}ms (max)'.format(
    np.mean(overhead_time)*1000, np.max(overhead_time)*1000))
print('Overhead percentage:      {:.1f}%'.format(
    100*np.mean(overhead_time)/np.mean(total_select_action_time)))

# Compare total_select_action_time with inf_time to find wrapper overhead
wrapper_overhead = np.mean(inf_time_fmpc_socp) - np.mean(total_select_action_time)
print('\nWrapper (experiment) overhead: {:.2f}ms ({:.1f}%)'.format(
    wrapper_overhead*1000, 100*wrapper_overhead/np.mean(inf_time_fmpc_socp)))
print('='*60)

print('\n--- MPC Warm-start Breakdown (averages) ---')
print('Block 1 - Solution Extraction:')
print('  State extraction (41 gets):  {:.2f}ms'.format(np.mean(mpc_state_extract_time)*1000))
print('  Input extraction (40 gets):  {:.2f}ms'.format(np.mean(mpc_input_extract_time)*1000))
print('  Total extraction 1:          {:.2f}ms'.format(np.mean(mpc_extract_time_1)*1000))
print('Block 1 - Logging (deepcopy):')
print('  deepcopy(x_prev):            {:.2f}ms'.format(np.mean(mpc_deepcopy_state_time)*1000))
print('  deepcopy(u_prev):            {:.2f}ms'.format(np.mean(mpc_deepcopy_input_time)*1000))
print('  deepcopy(goal_states):       {:.2f}ms'.format(np.mean(mpc_deepcopy_goal_time)*1000))
print('  Total logging 1:             {:.2f}ms'.format(np.mean(mpc_logging_time_1)*1000))
print('\nBlock 2 - DUPLICATE Extraction:')
print('  Total extraction 2:          {:.2f}ms'.format(np.mean(mpc_extract_time_2)*1000))
print('Block 2 - DUPLICATE Logging:')
print('  Total logging 2:             {:.2f}ms'.format(np.mean(mpc_logging_time_2)*1000))
print('\nSummary:')
total_warmstart_overhead = (np.mean(mpc_extract_time_1) + np.mean(mpc_logging_time_1) +
                             np.mean(mpc_extract_time_2) + np.mean(mpc_logging_time_2))
duplicate_waste = np.mean(mpc_extract_time_2) + np.mean(mpc_logging_time_2)
print('  Total warm-start overhead:   {:.2f}ms'.format(total_warmstart_overhead*1000))
print('  Duplicate waste:             {:.2f}ms ({:.1f}% of total)'.format(
    duplicate_waste*1000, 100*duplicate_waste/total_warmstart_overhead))
print('  Percentage of MPC total:     {:.1f}%'.format(
    100*total_warmstart_overhead/np.mean(mpc_total_time)))
print('='*60)

print('\nMaximum of inputs')
print('Thrust: {:.5f}N'.format(np.max(action_fmpc_socp[:, 0])))
print(' Angle: {:.2f}rad'.format(np.max(action_fmpc_socp[:, 1])))

vel_fmpc_socp = np.sqrt(state_fmpc_socp[:, 1]**2 + state_fmpc_socp[:, 3]**2)
print('\nVelocity on trajectory')
print('average velocity: {:.2f}m/s'.format(np.mean(vel_fmpc_socp)))
print('maximum velocity: {:.2f}m/s'.format(np.max(vel_fmpc_socp)))

print('\nDone!')
