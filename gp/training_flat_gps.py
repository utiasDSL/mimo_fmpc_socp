import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

from safe_control_gym.controllers.mpc.flat_gp_utils import GaussianProcess, ZeroMeanAffineGP

from scipy.spatial.distance import pdist, squareform

import pickle 
import os
import sys

from copy import deepcopy

def plot_data(states, time, title, label_x):
    '''plot states 
    input: states/inputs (array: [n_times, n_states])
            time (array: [n_times])'''
    nz = np.shape(states)[1]
    fig, axs = plt.subplots(nz)
    for k in range(nz):
        axs[k].plot(time,states[:, k], color='b', label=' ')
                
    axs[0].set_title(title)
    axs[-1].set(xlabel=label_x)

def plot_dist_matrix(dist_matrix, title=''):
    plt.figure()
    plt.imshow(dist_matrix, cmap='viridis', interpolation='nearest')  
    plt.colorbar(label="Distance")
    plt.title(title)
    plt.xlabel("Point Index")
    plt.ylabel("Point Index")

def plot_trained_gp(targets, means, preds, fig_count=0, show=False):
    lower, upper = preds.confidence_region()
    fig_count += 1
    confidence_size = upper.detach().numpy() - means.numpy()
    fig, ax = plt.subplots(2, 1) 
    ax[0].fill_between(list(range(lower.shape[0])), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, label='95%')
    ax[0].plot(means.squeeze(), 'r', label='GP Mean')
    ax[0].plot(targets.squeeze(), '*k', label='Targets')
    ax[0].legend()
    ax[0].set_title('Fitted GP')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('v')

    ax[1].set_title('2x standard deviation')
    ax[1].plot(confidence_size) 
    if show:
        plt.show()
    return fig_count

def train_gp(output_dir, inputs_train, targets_train, inputs_eval, targets_eval, PLOT=True):
    # Check if the folder exists, and create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder '{output_dir}' created.")
        
    # move to Torch
    inputs = torch.from_numpy(inputs_train)
    targets = torch.from_numpy(targets_train)

    train_in = inputs
    train_tar = targets

    # Setup GP
    gp_type = ZeroMeanAffineGP
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp = GaussianProcess(gp_type, likelihood, 1, output_dir)

    fname = os.path.join(output_dir, 'training_output.txt')
    orig_stdout = sys.stdout
    with open(fname,'w', 1) as print_to_file:
        sys.stdout = print_to_file
        loss_list = gp.train(train_in, train_tar.squeeze(), n_train=N_train, learning_rate=learning_rate)
    sys.stdout = orig_stdout

    if PLOT:
        fig, ax = plt.subplots(2)
        ax[0].plot(loss_list)
        ax[0].set_title('Loss')
        ax[1].plot(loss_list) 
        ax[1].set_title('Loss: logarithmic scale')
        ax[1].set_xscale('log')

    print('\n\nPrinting hyperparameters \n')
    print('Likelihood noise')
    print(gp.model.likelihood.noise.detach().numpy())
    print('kernel variance')
    print(gp.model.covar_module.variance.detach().numpy())
    print('kernel lengthscale')
    print(gp.model.covar_module.length.detach().numpy()[0:4])
    print(gp.model.covar_module.length.detach().numpy()[4:8])
    print(gp.model.covar_module.length.detach().numpy()[8:12])


    means, covs, preds = gp.predict(train_in)
    errors = means - train_tar.squeeze()
    abs_errors = torch.abs(errors)
    print('Training set mean error:', torch.mean(abs_errors).numpy())
    if PLOT:
        figcount = plot_trained_gp(train_tar, means, preds, 3)

    # test degredation of gammas
    means_from_gamma, cov_from_gamma, upper_from_gamma, lower_from_gamma  = gp.model.mean_and_cov_from_gammas(train_in)
    if PLOT:
        figcount+=1
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].plot(means_from_gamma, label='mean from gamma')
        ax[0, 0].plot(means, label='means predict')
        ax[0, 0].legend()
        ax[0, 0].set_title('Compare mean predict vs. mean from gamma')

        ax[0, 1].plot(cov_from_gamma, label='covs from gamma')
        ax[0, 1].plot(torch.diag(covs), label='covs predict')
        ax[0, 1].legend()
        ax[0, 1].set_title('Compare cov predict vs. cov from gamma')

        ax[1, 0].plot(means_from_gamma-means)
        ax[1, 0].set_title('Difference in means')

        ax[1, 1].plot(cov_from_gamma-torch.diag(covs))
        ax[1, 1].set_title('Difference in covs')

    # Show Quality on unseen data
    mean_eval, cov_eval, preds = gp.predict(inputs_eval)
    if PLOT:
        figcount = plot_trained_gp(targets_eval, mean_eval, preds, figcount)
    errors = mean_eval - targets_eval.squeeze()
    abs_errors = torch.abs(errors)
    print('Eval set mean error:', torch.mean(abs_errors).numpy())

    if PLOT:
        plt.show()

def prepare_data(inputs_raw, targets_raw, threshold, noise_std, normalization_vals, seed, PLOT):
    """
    filter based on similarity in normalized distance of [inputs, targets]
    add noise to targets with stddev noise_std
    normalize inputs with given normalization vector
    """

    # check similarity of training data with targets
    data_full_gp = np.hstack([inputs_raw, np.expand_dims(targets_raw, 1)])
    max_vals = np.max(np.abs(data_full_gp), axis=0)
    data_normalized_gp = data_full_gp/max_vals # normalize over all data, including targets
    dist_matrix = squareform(pdist(data_normalized_gp, metric='euclidean'))
    if PLOT:
        plot_dist_matrix(dist_matrix, "Pairwise Distance of full normalized training data")

    #remove data that is too similar
    filtered_indices = []
    for i in range(len(data_full_gp)):
        if all(dist_matrix[i, j] >= threshold for j in filtered_indices):
            filtered_indices.append(i)

    data_filtered_gp = data_full_gp[filtered_indices, :]
    data_filtered_gp_normalized = data_normalized_gp[filtered_indices, :] # just for visualization

    print(f'\n Number of datapoints remaining after downsampling: {len(filtered_indices)}\n')

    # check results of filter
    dist_matrix = squareform(pdist(data_filtered_gp_normalized, metric='euclidean'))
    if PLOT:
        plot_dist_matrix(dist_matrix, f"Normalized data filtered with distance threshold {threshold}")

    input_data = data_filtered_gp[:, :-1]
    targets_gp = data_filtered_gp[:, -1]

    np_rnd = np.random.default_rng(seed=seed)
    noise = np_rnd.normal(0, noise_std, size=targets_gp.shape)
    targets_gp_noisy = targets_gp + noise

    # normalize input data
    input_data = input_data/normalization_vals

    if PLOT:
        fig, ax = plt.subplots(1)
        ax.plot(targets_raw,'.', label='target data')
        ax.plot(filtered_indices, targets_gp,'.', label='similar points removed')
        ax.plot(filtered_indices, targets_gp_noisy, '.', label='with noise')
        ax.set_title(f'Training targets GP {do_gp_nr}')
        ax.legend()

        t = np.arange(0, np.shape(input_data)[0], 1)
        plot_data(input_data, t, 'Input data, similar points removed', 'index')
        plt.show()

    return input_data, targets_gp_noisy

######################################################################################################################
# Parameters
seed = 43
PLOT = True

# run from folder
training_data_file = './fgp/gp_train_data_noisyFig8.pkl'
eval_data_file = './fgp/gp_test_data.pkl' 

N_train = 2000 # number of training iterations in the GP
learning_rate = 0.02

noise_std_list = [2, 1.4] # for artificial noise
threshold = [0.25, 0.2]

# Training data
with open(training_data_file, 'rb') as file:
    train_data = pickle.load(file)
inputs_train_raw = train_data['inputs']
targets_train_raw = train_data['targets'] 

# Evaluation data
with open(eval_data_file, 'rb') as file:
    eval_data = pickle.load(file)
inputs_eval = eval_data['inputs']
inputs_eval = np.vstack(inputs_eval)
targets_eval = eval_data['targets'] 
targets_eval = np.vstack(targets_eval)

# remove position and velocity data, as the analytic transformation does not depend on it
rows_to_remove = [0, 1, 4, 5]
inputs_train_raw = np.delete(inputs_train_raw, rows_to_remove, axis=1)
inputs_eval = np.delete(inputs_eval, rows_to_remove, axis=1)

normalization_file_path = './fgp/normalization_arr.npy'
if False:
    normalization_vals = np.max(np.abs(inputs_train_raw), axis=0)
    print('saveing normalization values')
    print(normalization_vals)
    np.save(normalization_file_path, normalization_vals)
    exit()

normalization_vals = np.load(normalization_file_path)
print(f'Normalization vector: {normalization_vals}')

inputs_eval = inputs_eval/normalization_vals # normalize with same vector as before

##########
# Stuff specific to each GP 
do_gp_nr = 1 # which GP to train, 0 or 1

targets_raw_gp = targets_train_raw[:, do_gp_nr]
targets_eval = targets_eval[:, do_gp_nr] 

# data preparation: downsampling based on similarity, adding artificial noise
inputs_train, targets_train = prepare_data(inputs_train_raw, targets_raw_gp, threshold[do_gp_nr], noise_std_list[do_gp_nr], normalization_vals, seed, PLOT)

# GP training and testing
output_dir = f'./fgp/gp_v{do_gp_nr}'
train_gp(output_dir, inputs_train, targets_train, inputs_eval, targets_eval, PLOT)

