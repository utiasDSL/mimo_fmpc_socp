"""Minimal script to run gpmpc_acados_TP for quadrotor_2D_attitude tracking task."""

import os
import pickle
from collections import defaultdict
from functools import partial

import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, timing
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
from safe_control_gym.utils.gpmpc_plotting import make_quad_plots


@timing
def run_gpmpc_tp(gui=False, seed=1, save_data=True):
    """Run gpmpc_acados_TP controller learning and evaluation.

    Args:
        gui (bool): Whether to display the gui.
        seed (int): Random seed.
        save_data (bool): Whether to save the collected experiment data.
    """
    # Fixed configuration
    ALGO = 'gpmpc_acados_TP'
    SYS = 'quadrotor_2D_attitude'
    TASK = 'tracking'
    ADDITIONAL = '_10'
    PRIOR = '100'
    CTRL_ADD = ''

    # Check if config files exist
    task_config_path = f'./config_overrides/quadrotor_2D_attitude_tracking.yaml'
    algo_config_path = f'./config_overrides/gpmpc_acados_TP_training.yaml'

    assert os.path.exists(task_config_path), f'{task_config_path} does not exist'
    assert os.path.exists(algo_config_path), f'{algo_config_path} does not exist'

    # Setup configuration
    import sys
    sys.argv[1:] = [
        '--algo', ALGO,
        '--task', 'quadrotor',
        '--overrides',
        task_config_path,
        algo_config_path,
        #'--seed', str(seed),
        '--use_gpu', 'True',
        '--output_dir', f'./{ALGO}/results',
    ]

    # Create and merge config
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='train', help='main function to run.')
    fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
    config = fac.merge()

    # Setup output directory
    num_data_max = config.algo_config.num_epochs * config.algo_config.num_samples
    gp_tag = f'{PRIOR}_{num_data_max}'
    config.output_dir = os.path.join(config.output_dir, gp_tag + ADDITIONAL)
    set_dir_from_config(config)
    config.algo_config.output_dir = config.output_dir
    mkdirs(config.output_dir)

    # Set reward weights to match MPC weights
    config.task_config.rew_state_weight = config.algo_config.q_mpc
    config.task_config.rew_act_weight = config.algo_config.r_mpc

    # Create environment function
    env_func = partial(
        make,
        config.task,
        #seed=config.seed,
        **config.task_config
    )

    # Create random environment for initialization
    random_env = env_func(gui=False)

    # Create controller
    ctrl = make(
        config.algo,
        env_func,
        #seed=config.seed,
        **config.algo_config
    )

    # Get initial state
    init_state, _ = random_env.reset()

    # Create static environments
    static_env = env_func(gui=gui, randomized_init=False, init_state=init_state)
    static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)

    # Create experiment
    experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)

    # Training
    if config.algo_config.num_epochs == 1:
        print('Evaluating prior controller only (num_epochs=1)')
    elif config.algo_config.gp_model_path is not None:
        print(f'Loading GP model from {config.algo_config.gp_model_path}')
        ctrl.load(config.algo_config.gp_model_path)
    else:
        print('Starting controller learning...')
        experiment.reset()
        train_runs, test_runs = ctrl.learn(env=static_train_env)
        print('Controller learning complete')

        # Plot training results
        if isinstance(static_env, Quadrotor):
            make_quad_plots(
                test_runs=test_runs,
                train_runs=train_runs,
                trajectory=ctrl.traj.T,
                dir=ctrl.output_dir
            )

    # Evaluation
    print('Running evaluation...')
    trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1)

    # Close environments
    static_env.close()
    static_train_env.close()
    ctrl.close()
    random_env.close()

    # Compute metrics
    metrics = experiment.compute_metrics(trajs_data)

    # Save reference data
    ref_data = {
        'obs': trajs_data['obs'][0],
        'action': trajs_data['action'][0],
        'rmse': metrics['rmse'],
        'average_return': metrics['average_return'],
    }

    traj_length_str = str(config.task_config.episode_len_sec).replace('.', '_')
    np.save(
        f'./{config.output_dir}/{ALGO}_{SYS}_{traj_length_str}_ref_traj.npy',
        ref_data,
        allow_pickle=True
    )

    # Save results
    if save_data:
        results = {'trajs_data': trajs_data, 'metrics': metrics}
        with open(
            f'./{config.output_dir}/{config.algo}_data_{config.task}_{config.task_config.task}.pkl',
            'wb'
        ) as file:
            pickle.dump(results, file)

        # Save metrics to text file
        with open(f'./{config.output_dir}/metrics.txt', 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
        print(f'Metrics saved to ./{config.output_dir}/metrics.txt')

    # Plot evaluation results
    plot_quad_eval(results, experiment.env.env, config.output_dir)

    # Print final metrics
    print('\n' + '='*60)
    print('FINAL METRICS')
    print('='*60)
    for key, value in metrics.items():
        print(f'{key}: {value}')
    print('='*60)
    print(f'Final RMSE: {metrics["rmse"]:.4f} m')
    print(f'Final average return: {metrics["average_return"]:.4f}')

    return results, metrics


def plot_quad_eval(res, env, save_path=None):
    """Plot evaluation results for quadrotor tracking.

    Args:
        res (dict): Results dictionary containing trajs_data and metrics.
        env: Environment instance.
        save_path (str): Path to save plots.
    """
    state_stack = res['trajs_data']['obs'][0]
    input_stack = res['trajs_data']['action'][0]
    constraint_stack = [
        res['trajs_data']['info'][0][i]['constraint_values']
        for i in range(1, len(res['trajs_data']['info'][0]))
    ]
    constraint_stack = np.array(constraint_stack)
    model = env.symbolic

    if env.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
        x_idx, z_idx = 0, 2
    elif env.QUAD_TYPE in [QuadType.THREE_D_ATTITUDE,
                           QuadType.THREE_D_ATTITUDE_10,
                           QuadType.THREE_D_ATTITUDE_DELAY]:
        x_idx, y_idx, z_idx = 0, 2, 4

    stepsize = model.dt
    plot_length = min(input_stack.shape[0], state_stack.shape[0])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    reference = env.X_GOAL
    if env.TASK == Task.STABILIZATION:
        reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))
    action_bound = env.action_space

    # Plot states
    fig, axs = plt.subplots(model.nx, figsize=(8, model.nx * 1))
    for k in range(model.nx):
        axs[k].plot(times, state_stack.T[k, :plot_length], label='actual')
        axs[k].plot(times, reference.T[k, :plot_length], color='r', label='desired')
        axs[k].set(ylabel=env.STATE_LABELS[k] + f'\n[{env.STATE_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if k != model.nx - 1:
            axs[k].set_xticks([])
    axs[0].set_title('State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'state_trajectories.png'))

    # Plot inputs
    _, axs = plt.subplots(model.nu, figsize=(8, model.nu * 1))
    if model.nu == 1:
        axs = [axs]
    for k in range(model.nu):
        axs[k].plot(times, input_stack.T[k, :plot_length])
        axs[k].set(ylabel=f'input {k}')
        axs[k].hlines(action_bound.high[k], 0, times[-1], color='gray', linestyle='--')
        axs[k].hlines(action_bound.low[k], 0, times[-1], color='gray', linestyle='--')
        axs[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[0].set_title('Input Trajectories')
    axs[-1].set(xlabel='time (sec)')
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'input_trajectories.png'))

    # Plot state path in x-z plane with tracking error
    fig, axs = plt.subplots(2, figsize=(8, 8))
    axs[0].plot(state_stack.T[x_idx, :plot_length],
                state_stack.T[z_idx, :plot_length], label='actual')
    axs[0].plot(reference.T[x_idx, :plot_length],
                reference.T[z_idx, :plot_length], color='r', label='desired')
    axs[0].set_xlabel('x [m]')
    axs[0].set_ylabel('z [m]')
    axs[0].set_title('State path in x-z plane')
    axs[0].legend()

    error = np.array([
        np.sqrt(res['trajs_data']['info'][0][i]['mse'])
        for i in range(1, len(res['trajs_data']['info'][0]))
    ])
    rmse = res['metrics']['rmse']

    axs[1].plot(times, error)
    axs[1].set_xlabel('time [s]')
    axs[1].set_ylabel('tracking error [m]')
    axs[1].set_title(f'Tracking error {rmse:.4f} m')

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'state_xz_path.png'))
        print(f'Plots saved to {save_path}')

    # Plot x-y plane for 3D quadrotors
    if env.QUAD_TYPE in [QuadType.THREE_D_ATTITUDE,
                         QuadType.THREE_D_ATTITUDE_10]:
        fig, axs = plt.subplots(1)
        axs.plot(state_stack.T[x_idx, :plot_length],
                 state_stack.T[y_idx, :plot_length], label='actual')
        axs.plot(reference.T[x_idx, :plot_length],
                 reference.T[y_idx, :plot_length], color='r', label='desired')
        axs.set_xlabel('x [m]')
        axs.set_ylabel('y [m]')
        axs.set_title('State path in x-y plane')
        axs.legend()
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'state_xy_path.png'))

    # Plot constraint violations
    fig, axs = plt.subplots(len(constraint_stack[0]), figsize=(8, len(constraint_stack[0]) * 1))
    constr_state_idx = 0
    for k in range(len(constraint_stack[0])):
        axs[k].plot(times, constraint_stack[:, k], label='actual')
        violated = np.where(constraint_stack[:, k] > 0, 1, 0)
        violated_step = np.where(violated == 1)
        violated_values = constraint_stack[violated_step, k]
        axs[k].scatter(times[violated_step], violated_values,
                       color='red', label='violated', marker='x')

        constr_state_idx += 1 if k % 2 == 0 else 0
        if constr_state_idx - 1 < len(env.STATE_LABELS):
            axs[k].set(ylabel=f'constraint {constr_state_idx - 1}' +
                      f'\n[{env.STATE_UNITS[constr_state_idx - 1]}]')
        else:
            axs[k].set(ylabel=f'constraint {constr_state_idx - 1}' +
                      f'\n[{env.ACTION_UNITS[constr_state_idx - 1 - len(env.STATE_LABELS)]}]')

        axs[k].hlines(0, 0, times[-1], color='gray', linestyle='--')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    axs[0].set_title('Constraint Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure,
                   bbox_to_anchor=(1, 0), loc='upper right')
    axs[-1].set(xlabel='time (sec)')
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'constraint_trajectories.png'))

    # Plot individual x, y, z tracking errors for 3D quadrotors
    if env.QUAD_TYPE in [QuadType.THREE_D_ATTITUDE,
                         QuadType.THREE_D_ATTITUDE_10]:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        x_error = np.abs(state_stack.T[x_idx, :plot_length] - reference.T[x_idx, :plot_length])
        y_error = np.abs(state_stack.T[y_idx, :plot_length] - reference.T[y_idx, :plot_length])
        z_error = np.abs(state_stack.T[z_idx, :plot_length] - reference.T[z_idx, :plot_length])

        axs[0, 0].plot(times, x_error)
        axs[0, 0].set_xlabel('time [s]')
        axs[0, 0].set_ylabel('x tracking error [m]')
        axs[0, 0].set_title(f'X Tracking Error (RMSE: {np.sqrt(np.mean(x_error ** 2)):.4f} m)')

        axs[0, 1].plot(times, y_error)
        axs[0, 1].set_xlabel('time [s]')
        axs[0, 1].set_ylabel('y tracking error [m]')
        axs[0, 1].set_title(f'Y Tracking Error (RMSE: {np.sqrt(np.mean(y_error ** 2)):.4f} m)')

        axs[1, 0].plot(times, z_error)
        axs[1, 0].set_xlabel('time [s]')
        axs[1, 0].set_ylabel('z tracking error [m]')
        axs[1, 0].set_title(f'Z Tracking Error (RMSE: {np.sqrt(np.mean(z_error ** 2)):.4f} m)')

        combined_error = np.sqrt(x_error ** 2 + y_error ** 2 + z_error ** 2)
        axs[1, 1].plot(times, combined_error)
        axs[1, 1].set_xlabel('time [s]')
        axs[1, 1].set_ylabel('combined tracking error [m]')
        axs[1, 1].set_title(f'Combined Tracking Error (RMSE: {np.sqrt(np.mean(combined_error ** 2)):.4f} m)')

        fig.tight_layout()

        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'xyz_tracking_errors.png'))


if __name__ == '__main__':
    results, metrics = run_gpmpc_tp()
