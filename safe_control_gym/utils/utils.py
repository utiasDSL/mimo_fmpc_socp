'''Miscellaneous utility functions.'''

import argparse
import datetime
import json
import os
import random
import subprocess
import sys
from functools import wraps
import time

import gymnasium as gym
import imageio
import munch
import numpy as np
import torch
import yaml
from termcolor import colored

import pandas as pd
from enum import IntEnum, unique
import numpy as np
from scipy.signal import butter

GRAVITY = 9.81

def mkdirs(*paths):
    '''Makes a list of directories.'''

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def eval_token(token):
    '''Converts string token to int, float or str.'''
    if token.isnumeric():
        return int(token)
    try:
        return float(token)
    except TypeError:
        return token


def read_file(file_path, sep=','):
    '''Loads content from a file (json, yaml, csv, txt).

    For json & yaml files returns a dict.
    Ror csv & txt returns list of lines.
    '''
    if len(file_path) < 1 or not os.path.exists(file_path):
        return None
    # load file
    f = open(file_path, 'r')
    if 'json' in file_path:
        data = json.load(f)
    elif 'yaml' in file_path:
        data = yaml.load(f, Loader=yaml.FullLoader)
    else:
        sep = sep if 'csv' in file_path else ' '
        data = []
        for line in f.readlines():
            line_post = [eval_token(t) for t in line.strip().split(sep)]
            # if only sinlge item in line
            if len(line_post) == 1:
                line_post = line_post[0]
            if len(line_post) > 0:
                data.append(line_post)
    f.close()
    return data


def merge_dict(source_dict, update_dict):
    '''Merges updates into source recursively.'''
    for k, v in update_dict.items():
        if k in source_dict and isinstance(source_dict[k], dict) and isinstance(
                v, dict):
            merge_dict(source_dict[k], v)
        else:
            source_dict[k] = v


def get_time():
    '''Gets current timestamp (as string).'''
    start_time = datetime.datetime.now()
    time = str(start_time.strftime('%Y_%m_%d-%X'))
    return time


def get_random_state():
    '''Snapshots the random state at any moment.'''
    return {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state()
    }


def set_random_state(state_dict):
    '''Resets the random state for experiment restore.'''
    random.setstate(state_dict['random'])
    np.random.set_state(state_dict['numpy'])
    torch.torch.set_rng_state(state_dict['torch'])


def set_seed(seed, cuda=False):
    '''General seeding function for reproducibility.'''
    assert seed is not None, 'Error in set_seed(...), provided seed not valid'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_dir_from_config(config):
    '''Creates a output folder for experiment (and save config files).

    Naming format: {root (e.g. results)}/{tag (exp id)}/{seed}_{timestamp}_{git commit id}
    '''
    # Make run folder (of a seed run for an experiment)
    seed = str(config.seed) if config.seed is not None else '-'
    timestamp = str(datetime.datetime.now().strftime('%b-%d-%H-%M-%S'))
    try:
        commit_id = subprocess.check_output(
            ['git', 'describe', '--tags', '--always']
        ).decode('utf-8').strip()
        commit_id = str(commit_id)
    except BaseException:
        commit_id = '-'
    run_dir = f'seed{seed}_{timestamp}_{commit_id}'
    # Make output folder.
    config.output_dir = os.path.join(config.output_dir, config.tag, run_dir)
    mkdirs(config.output_dir)
    # Save config.
    with open(os.path.join(config.output_dir, 'config.yaml'), 'w') as file:
        yaml.dump(munch.unmunchify(config), file, default_flow_style=False)
    # Save command.
    with open(os.path.join(config.output_dir, 'cmd.txt'), 'a') as file:
        file.write(' '.join(sys.argv) + '\n')


def set_seed_from_config(config):
    '''Sets seed, only set if seed is provided.'''
    seed = config.seed
    if seed is not None:
        set_seed(seed, cuda=config.use_gpu)


def set_device_from_config(config):
    '''Sets device, using GPU is set to `cuda` for now, no specific GPU yet.'''
    use_cuda = config.use_gpu and torch.cuda.is_available()
    config.device = 'cuda' if use_cuda else 'cpu'


def save_video(name, frames, fps=20):
    '''Convert list of frames (H,W,C) to a video.

    Args:
        name (str): path name to save the video.
        frames (list): frames of the video as list of np.arrays.
        fps (int, optional): frames per second.
    '''
    assert '.gif' in name or '.mp4' in name, 'invalid video name'
    vid_kwargs = {'fps': fps}
    h, w, c = frames[0].shape
    video = np.stack(frames, 0).astype(np.uint8).reshape(-1, h, w, c)
    imageio.mimsave(name, video, **vid_kwargs)


def str2bool(val):
    '''Converts a string into a boolean.

    Args:
        val (str|bool): Input value (possibly string) to interpret as boolean.

    Returns:
        bool: Interpretation of `val` as True or False.
    '''
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('[ERROR] in str2bool(), a Boolean value is expected')


def unwrap_wrapper(env, wrapper_class):
    '''Retrieve a ``VecEnvWrapper`` object by recursively searching.'''
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env, wrapper_class):
    '''Check if a given environment has been wrapped with a given wrapper.'''
    return unwrap_wrapper(env, wrapper_class) is not None

def timing(f):
    '''Decorator for measuring the time of a function.
       The elapsed time is stored in the function object.
       Only prints if self.verbose=True (checks first argument for verbose attribute)
    '''
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        wrap.elapsed_time = te - ts
        # Only print if verbose flag is True (check first argument for verbose attribute)
        if len(args) > 0 and hasattr(args[0], 'verbose') and args[0].verbose:
            print(colored(f'func:{f.__name__} took: {wrap.elapsed_time:.4f} sec', 'blue' ))
        return result
    return wrap






@unique
class DataVarIndex(IntEnum):
    '''A class that creates ids for the data.'''

    TIME = 0
    POS_X = 1
    POS_Y = 2
    POS_Z = 3
    ROLL = 4
    PITCH = 5
    YAW = 6
    VEL_X = 7
    VEL_Y = 8
    VEL_Z = 9
    ROLL_RATE = 10
    PITCH_RATE = 11
    YAW_RATE = 12
    CMD_ROLL = 13
    CMD_PITCH = 14
    CMD_YAW = 15
    CMD_THRUST = 16
    DES_POS_X = 17
    DES_POS_Y = 18
    DES_POS_Z = 19
    DES_YAW = 20
    DES_VEL_X = 21
    DES_VEL_Y = 22
    DES_VEL_Z = 23
    STATUS = 24  # One of the following: "TAKEOFF", "LAND", "TRACK_TRAJ"
    VICON_POS_X = 25
    VICON_POS_Y = 26
    VICON_POS_Z = 27
    VICON_ROLL = 28
    VICON_PITCH = 29
    VICON_YAW = 30
    ACC_X = 31
    ACC_Y = 32
    ACC_Z = 33
    ROLL_ACC = 34
    PITCH_ACC = 35
    YAW_ACC = 36
    PITCH_PRED = 37
    PITCH_ACT = 38
    OBS_POS_X = 39
    OBS_POS_Z = 40
    OBS_VEL_X = 41
    OBS_VEL_Z = 42
    OBS_PITCH = 43
    OBS_PITCH_RATE = 44
    INFERENCE_TIME = 45
    POS_ERROR = 46
    VICON_VEL_X = 47
    VICON_VEL_Y = 48
    VICON_VEL_Z = 49
    VICON_ROLL_RATE = 50
    VICON_PITCH_RATE = 51
    VICON_YAW_RATE = 52
    CMD_FORCE = 53


var_bounds = {
    DataVarIndex.CMD_THRUST: (2.0e4, 65535.0),
}


@unique
class Status(IntEnum):
    '''A class that creates ids for the status of the drone.'''
    TAKEOFF = 0
    LAND = 1
    TRACK_TRAJ = 2
    HOVER = 3
    VERTICAL = 4
    INTERPOLATE = 5
    HORIZONTAL = 6


match_status = {
        Status.TAKEOFF.name: Status.TAKEOFF,
        Status.LAND.name: Status.LAND,
        Status.TRACK_TRAJ.name: Status.TRACK_TRAJ,
        Status.HOVER.name: Status.HOVER,
        Status.VERTICAL.name: Status.VERTICAL,
        Status.INTERPOLATE.name: Status.INTERPOLATE,
        Status.HORIZONTAL.name: Status.HORIZONTAL,
        }


def load_data(filename):
    """Load the data from the csv file and return it as a numpy array."""
    # Read the data from the csv file, skipping the first row
    # and the last column has to be transformed using Status enum
    pd_data = pd.read_csv(filename)
    pd_data[DataVarIndex.STATUS.name] = pd_data[DataVarIndex.STATUS.name].apply(lambda s: match_status[s])
    data = pd_data.to_numpy()

    # There may be a mismatch in the number of columns and the number of DataVarIndex. Add dummy values for the missing columns
    num_columns = len(DataVarIndex)
    num_data_columns = data.shape[1]
    dummy_data = np.zeros((data.shape[0], num_columns - num_data_columns))
    data = np.hstack((data, dummy_data))

    return data


def get_file_path_from_run(wandb_project, run_name=None, file_name=None, use_latest=False, smoothed=False):
    """
    Get the file path from the specified run name or file name. 
    Run name has higher priority than file name.
    """
    # Use the latest run if no run name or file name is provided
    if run_name is None and file_name is None:
        use_latest = True

    wandb_api = wandb.Api()
    runs = wandb_api.runs(wandb_project, order='-created_at')
    
    if use_latest:
        # Get the latest run
        run = runs[0]
    elif run_name is not None:
        # Get the run with the specified name
        run = None
        for r in runs:
            if r.name == run_name:
                run = r
                break
        if run is None:
            raise ValueError("Run with name {} not found".format(run_name))
    elif file_name is not None:
        # Get the run with the specified file name
        run = None
        for r in runs:
            r_json = json.loads(r.json_config)
            run_file_name = r_json['file_path']['value'].rsplit('/')[-1]
            print(run_file_name, file_name)
            if run_file_name == file_name:
                run = r
                break
        if run is None:
            raise ValueError("Run with file name {} not found".format(file_name))
    else:
        raise ValueError("No run name or file name provided")
    
    run_json = json.loads(run.json_config)
    print("Run name: ", run.name)

    file_path = run_json['file_path']['value']
    traj_plane = run_json['traj_plane']['value']
    algo = run_json['algo']['value'] if run_json.get('algo') is not None else None
    task = run_json['task']['value'] if run_json.get('task') is not None else 'tracking'
    # algo = getattr(run_json, 'algo', 'None')
    # task = getattr(run_json, 'task', 'tracking') 

    config_info = {'run_name': run.name, 
                   'algo': algo,
                   'is_real': run_json['is_real']['value'],
                   'task': task,
                   }
    return file_path, traj_plane, config_info


def scipy_lowpass(cutoff_freq, 
                  sample_time, 
                  array_cur, 
                  array_unfiltered, 
                  array_filtered, 
                  order=4):
    """Apply a lowpass filter online to the data using scipy.
    
    Args:
        cutoff_freq (float): cutoff frequency of the filter. Must be less than half the sample rate.
        sample_time (float): sample time of the data stream.
        array_cur (np.array, dim_state): current data.
        array_unfiltered_hist (np.array, order x dim_state): unfiltered data in the past
        array_filtered_hist (np.array, order x dim_state): filtered data in the past
        order (int): order of the filter.
    """
    assert cutoff_freq < 0.5 / sample_time, "Cutoff frequency must be less than half the sample rate."
    sample_rate = 1.0 / sample_time
    nyquist_freq = 0.5 * sample_rate
    # nyquist normalized cutoff for digital design
    Wn = cutoff_freq / nyquist_freq
    # b, a = butter(4, Wn, btype='lowpass')
    b, a = butter(order, Wn, btype='lowpass')
    # print('a: ', a)
    # print('b: ', b)
    result = b[0] * array_cur
    for i in range(order):
        result += -a[i+1]*array_filtered[i, :] + b[i+1]*array_unfiltered[i, :]
    return result