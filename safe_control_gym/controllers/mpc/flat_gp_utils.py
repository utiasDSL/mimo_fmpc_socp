import numpy as np
import gpytorch
import torch
import os
import matplotlib.pyplot as plt
from gpytorch.constraints import Positive

def weighted_distance(x1 : torch.Tensor, x2 : torch.Tensor, L : torch.Tensor) -> torch.Tensor:
    """Computes (x1-x2)^T L (x1-x2)
    Args:
        x1 (torch.tensor) : N x n tensor (N is number of samples and n is dimension of vector)
        x2 (torch.tensor) : M x n tensor
        L (torch.tensor) : nxn wieght matrix

    Returns:
        weighted_distances (torch.tensor) : N x M tensor of all the prossible products

    """
    N = x1.shape[0]
    M = x2.shape[0]
    n = L.shape[0]
    # subtract all vectors in x2 from all vectors in x1 (results in NxMxn matrix)
    diff = x1.unsqueeze(1) - x2
    diff_T = diff.reshape(N,M,n,1)
    diff = diff.reshape(N,M,1,n)
    L = L.reshape(1,1,n,n)
    L = L.repeat(N,M,1,1)
    weighted_distances = torch.matmul(diff, torch.matmul(L,diff_T))
    return weighted_distances.squeeze()

def squared_exponential(x1,x2,L,var):
    return var * torch.exp(-0.5 * weighted_distance(x1, x2, L))

class AffineKernel(gpytorch.kernels.Kernel):
    is_stationary = False
    def __init__(self, input_dim,
                 length_prior=None,
                 length_constraint=None,
                 variance_prior=None,
                 variance_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.zeros(3*(self.input_dim-2)))
        )
        self.register_parameter(
            name='raw_variance', parameter=torch.nn.Parameter(torch.zeros(3))
        )
        # set the parameter constraint to be positive, when nothing is specified
        if length_constraint is None:
            length_constraint = Positive()
        if variance_constraint is None:
            variance_constraint = Positive()
        # register the constraints
        self.register_constraint("raw_length", length_constraint)
        self.register_constraint("raw_variance", variance_constraint)
        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if length_prior is not None:
            self.register_prior(
                "length_prior",
                length_prior,
                lambda m: m.length,
                lambda m, v: m._set_length(v),
            )
        if variance_prior is not None:
            self.register_prior(
                "variance_prior",
                variance_prior,
                lambda m: m.variance,
                lambda m, v: m._set_variance(v),
            )

    # now set up the 'actual' paramter
    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    def _set_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))


    @property
    def variance(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        return self._set_variance(value)

    def _set_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_variance)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_variance=self.raw_variance_constraint.inverse_transform(value))

    def kappa_alpha(self,x1, x2):
        L_alpha = torch.diag(1 / (self.length[0:self.input_dim-2] ** 2))
        var_alpha = self.variance[0]
        return squared_exponential(x1, x2, L_alpha, var_alpha)

    def kappa_beta1(self, x1, x2):
        L_beta = torch.diag(1 / (self.length[self.input_dim-2:2*(self.input_dim-2)] ** 2))
        var_beta = self.variance[1]
        return squared_exponential(x1, x2, L_beta, var_beta)
    
    def kappa_beta2(self, x1, x2):
        L_beta = torch.diag(1 / (self.length[2*(self.input_dim-2):] ** 2))
        var_beta = self.variance[2]
        return squared_exponential(x1, x2, L_beta, var_beta)

    def kappa(self,x1, x2):
        z1 = x1[:, 0:self.input_dim - 2]
        u1 = x1[:, -2:]
        z2 = x2[:, 0:self.input_dim - 2]
        u2 = x2[:, -2:]           
        u_mat_1 = u1[:, 0].unsqueeze(1) * u2[:, 0] # note: some extra dims removed from SISO code, seems to work though
        u_mat_2 = u1[:, 1].unsqueeze(1) * u2[:, 1] 
        kappa = self.kappa_alpha(z1, z2) + self.kappa_beta1(z1, z2) * u_mat_1.squeeze() + self.kappa_beta2(z1, z2) * u_mat_2.squeeze()
        if kappa.dim() < 2:
            return kappa.unsqueeze(0)
        else:
            return kappa
    # this is the kernel function
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
        kern = self.kappa(x1, x2)
        if diag:
            return kern.diag()
        else:
            return kern

class AffineGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        """Zero mean with Affine Kernel GP model for SISO systems

        Args:
            train_x (torch.Tensor): input training data (N_samples x input_dim)
            train_y (torch.Tensor): output training data (N_samples x 1)
            likelihood (gpytorch.likelihood): Likelihood function
                (gpytorch.likelihoods.MultitaskGaussianLikelihood)
        """
        super().__init__(train_x, train_y, likelihood)
        self.input_dim = train_x.shape[1]
        #self.output_dim = train_y.shape[1]
        self.output_dim = 1
        self.n = 1     #output dimension
        #self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = None
        self.covar_module = AffineKernel(self.input_dim)
        self.K_plus_noise_inv = None

    def forward(self, x):
        mean_x = self.mean_module(x) # is this needed for ZeroMean?
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def compute_gammas(self, query):
        return NotImplementedError

    def mean_and_cov_from_gammas(self,query):
        gamma_1, gamma_2, gamma_3, gamma_4, gamma_5 = self.compute_gammas(query)
        u = query[:, -2:, None]
        # compute transposes
        gamma_2_trans = torch.transpose(gamma_2, 1, 2)
        gamma_4_trans = torch.transpose(gamma_4, 1, 2)
        u_trans = torch.transpose(u, 1, 2)

        means_from_gamma = gamma_1 + gamma_2_trans @ u
        covs_from_gamma = gamma_3 + gamma_4_trans @ u + u_trans @ gamma_5 @ u  + self.likelihood.noise.detach()
        upper_from_gamma = means_from_gamma + covs_from_gamma.sqrt() * 2
        lower_from_gamma = means_from_gamma - covs_from_gamma.sqrt() * 2
        return means_from_gamma.squeeze(), covs_from_gamma.squeeze(), upper_from_gamma.squeeze(), lower_from_gamma.squeeze()

class ZeroMeanAffineGP(AffineGP):
    def __init__(self, train_x, train_y, likelihood):
        """Zero mean with Affine Kernel GP model for SISO systems

        Args:
            train_x (torch.Tensor): input training data (N_samples x input_dim)
            train_y (torch.Tensor): output training data (N_samples x 1)
            likelihood (gpytorch.likelihood): Likelihood function
                (gpytorch.likelihoods.MultitaskGaussianLikelihood)
        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

    def compute_gammas(self, query):
        # Parse inputs
        with torch.no_grad():
            n_train_samples = self.train_targets.shape[0]
            n_query_samples = query.shape[0]
            zq = query[:,0:self.input_dim-2]
            # uq = query[:,-2,None] # do not need it here
            z_train = self.train_inputs[0][:,0:self.input_dim-2]
            u_train = self.train_inputs[0][:,-2:]

            # Precompute useful matrices
            k_a = self.covar_module.kappa_alpha(zq, z_train)
            k_b1 = self.covar_module.kappa_beta1(zq, z_train)            
            k_b2 = self.covar_module.kappa_beta2(zq, z_train)
            if n_query_samples == 1:
                k_a = k_a.unsqueeze(0)
                k_b1 = k_b1.unsqueeze(0)
                k_b2 = k_b2.unsqueeze(0)
            k_a = k_a.unsqueeze(1) 
            k_beta = torch.stack([k_b1, k_b2], 1)                

            u_train_mat = (u_train.T).tile([n_query_samples, 1, 1])

            k_b = torch.mul(k_beta, u_train_mat)
            
            k_a_trans = torch.transpose(k_a, 1, 2) # transpose in the second two dimensions, keeps batch at front

            # first part of gamma5 correctly with batch as first dim
            k_beta12_diag = torch.zeros(n_query_samples, 2, 2)      
            k_beta12_diag[:, 0, 0] = torch.diag(torch.atleast_2d(self.covar_module.kappa_beta1(zq, zq)))
            k_beta12_diag[:, 1, 1] = torch.diag(torch.atleast_2d(self.covar_module.kappa_beta2(zq, zq)))

            Psi = self.train_targets.reshape((n_train_samples,1)) # equal to train_targets.transpose()
            # compute gammas
            gamma_1 = k_a @ self.K_plus_noise_inv @ Psi
            gamma_2 = k_b @ self.K_plus_noise_inv @ Psi             
            gamma_3 = torch.diag(torch.atleast_2d(self.covar_module.kappa_alpha(zq,zq))).unsqueeze(1).unsqueeze(2) \
                - (k_a @ self.K_plus_noise_inv @ k_a_trans)
            gamma_4 = -2*(k_b @ self.K_plus_noise_inv @ k_a_trans)
            gamma_5 = k_beta12_diag - k_b @ self.K_plus_noise_inv @ torch.transpose(k_b, 1, 2)
        return gamma_1, gamma_2, gamma_3, gamma_4, gamma_5 # nothing squeezed, all in full dimension n_query_samples x ** x **

class GaussianProcess():
    def __init__(self, model_type, likelihood, n, save_dir):
        """
        Gaussian Process decorator for gpytorch
        Args:
            model_type (gpytorch model class): Model class for the GP (ZeroMeanIndependentMultitaskGPModel)
            likelihood (gpytorch.likelihood): likelihood function
            n (int): Dimension of input state space

        """
        self.model_type = model_type
        self.likelihood = likelihood
        self.m = n
        self.optimizer = None
        self.model = None
        self.save_dir = save_dir


    def init_with_hyperparam(self,
                            path_to_model,
                             train_inputs=None,
                             train_targets=None
                             ):
        device = torch.device('cpu')
        fname_sd = os.path.join(path_to_model, 'model.pth')
        state_dict = torch.load(fname_sd, map_location=device)
        if train_targets is None or train_inputs is None:
            fname_data = os.path.join(path_to_model, 'train_data.pt')
            data = torch.load(fname_data)
            if train_inputs is None:
                train_inputs = data['inputs']
            if train_targets is None:
                train_targets = data['targets']
        if self.model is None:
            self.model = self.model_type(train_inputs,
                                         train_targets,
                                         self.likelihood)

        self.model.load_state_dict(state_dict)
        self.model.double() # needed
        self._compute_GP_covariances(train_inputs)

    def _compute_GP_covariances(self,
                                train_x
                                ):
        """Compute K(X,X) + sigma*I and its inverse.

        """
        # Pre-compute inverse covariance plus noise to speed-up computation.
        K_lazy = self.model.covar_module(train_x.double())
        K_lazy_plus_noise = K_lazy.add_diag(self.model.likelihood.noise)
        n_samples = train_x.shape[0]
        self.model.K_plus_noise_inv = K_lazy_plus_noise.inv_matmul(torch.eye(n_samples).double())

        # for testing only: compute condition number of K + noise

        cond_number = torch.linalg.cond(K_lazy_plus_noise.evaluate()).item()
        print(f"Condition number of the covariance matrix + noise: {cond_number:.2e}")

    def train(self, train_x, train_y, n_train=150, learning_rate=0.01, gpu=True):
        """
        Train the GP using Train_x and Train_y
        train_x: Torch tensor (dim input x N samples)
        train_y: Torch tensor (nx x N samples)

        """
        self.n = train_x.shape[1]
        self.m = 1
        self.output_dim = 1
        self.input_dim = train_x.shape[1]
        if self.model is None:
            self.model = self.model_type(train_x, train_y, self.likelihood)
        else:
            train_x = torch.reshape(train_x, self.model.train_inputs[0].shape)
            train_x = torch.cat([train_x, self.model.train_inputs[0]])
            train_y = torch.cat([train_y, self.model.train_targets])
            self.model.set_train_data(train_x, train_y, False)
        if gpu:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            self.model.cuda()
            self.likelihood.cuda()

        self.model.double()
        self.likelihood.double()
        self.model.train()
        self.likelihood.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        loss_list =[]
        for i in range(n_train):
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, n_train, loss.item()))
            loss_list.append(loss.item())
            self.optimizer.step()

        # compute inverse covariance plus noise for faster computation later
        self.model = self.model.cpu()
        self.likelihood = self.likelihood.cpu()
        train_x = train_x.cpu()
        train_y = train_y.cpu()
        state_dict = self.model.state_dict()
        fname = os.path.join(self.save_dir, 'model.pth')
        torch.save(state_dict, fname)
        data = {'inputs': train_x, 'targets': train_y}
        torch.save(data, os.path.join(self.save_dir, 'train_data.pt'))
        self._compute_GP_covariances(train_x)
        return loss_list

    def predict(self, x, requires_grad=False, return_pred=True):
        """
        x : torch.Tensor (input dim X N_samples)
        Return
            Predicitons
            mean : torch.tensor (nx X N_samples)
            cov
            pred (if flag is true)
        """
        #x = torch.from_numpy(x).double()
        self.model.eval()
        self.likelihood.eval()
        #with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).double()

        if requires_grad:
            predictions = self.likelihood(self.model(x))
            mean = predictions.mean
            cov = predictions.covariance_matrix
        else:
            with torch.no_grad():
                predictions = self.likelihood(self.model(x))
                mean = predictions.mean
                cov = predictions.covariance_matrix
        if return_pred:
            return mean, cov, predictions
        else:
            return mean, cov

    def plot_trained_gp(self, t, fig_count=0):
        means, covs, preds = self.predict(self.model.train_inputs[0])
        lower, upper = preds.confidence_region()
        fig_count += 1
        plt.figure(fig_count)
        plt.fill_between(t, lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, label='95%')
        plt.plot(t, means, 'r', label='GP Mean')
        plt.plot(t, self.model.train_targets, '*k', label='Data')
        plt.legend()
        plt.title('Fitted GP')
        plt.xlabel('Time (s)')
        plt.ylabel('v')
        plt.show()

        return fig_count


class FastGPNumpy:
    """
    Efficient numpy-based GP inference for ZeroMeanAffineGP.
    Precomputes all static matrices for fast gamma computation at query time.

    This class extracts all trained parameters from a GPyTorch GaussianProcess
    and performs inference entirely in numpy, avoiding device transfers and
    autograd overhead.
    """

    def __init__(self, gpytorch_gp_wrapper):
        """
        Extract and precompute all static quantities from trained GaussianProcess.

        Args:
            gpytorch_gp_wrapper (GaussianProcess): Trained GP wrapper with ZeroMeanAffineGP model
        """
        model = gpytorch_gp_wrapper.model

        # Extract training data
        with torch.no_grad():
            # Full training data
            X_train = model.train_inputs[0].cpu().numpy()  # (n_samples, input_dim)
            self.y_train = model.train_targets.cpu().numpy().reshape(-1, 1)  # (n_samples, 1)

            # Split into state and control parts
            self.input_dim = X_train.shape[1]
            self.z_train = X_train[:, :self.input_dim - 2]  # (n_samples, input_dim-2)
            self.u_train = X_train[:, -2:]  # (n_samples, 2)
            self.n_train = self.z_train.shape[0]

            # Extract kernel hyperparameters
            kernel = model.covar_module
            length = kernel.length.cpu().numpy()
            variance = kernel.variance.cpu().numpy()

            # Separate hyperparameters for alpha, beta1, beta2 kernels
            n_state_dim = self.input_dim - 2
            self.lengthscales_alpha = length[0:n_state_dim]
            self.lengthscales_beta1 = length[n_state_dim:2*n_state_dim]
            self.lengthscales_beta2 = length[2*n_state_dim:]

            self.var_alpha = variance[0]
            self.var_beta1 = variance[1]
            self.var_beta2 = variance[2]

            # Extract noise variance
            self.noise_var = gpytorch_gp_wrapper.likelihood.noise.cpu().numpy().item()

            # Extract precomputed K_inv (already computed by GaussianProcess._compute_GP_covariances)
            self.K_inv = model.K_plus_noise_inv.cpu().numpy()  # (n_samples, n_samples)

            # Precompute K_inv @ y_train for mean predictions
            self.K_inv_y = self.K_inv @ self.y_train  # (n_samples, 1)

    def _compute_sq_exp_kernel(self, z1, z2, lengthscales, variance):
        """
        Compute squared exponential kernel between z1 and z2.

        k(z1, z2) = var * exp(-0.5 * sum((z1 - z2)^2 / lengthscales^2))

        Args:
            z1: numpy array (d,) - single query point
            z2: numpy array (n, d) - training points
            lengthscales: numpy array (d,) - length scales for each dimension
            variance: float - output variance

        Returns:
            numpy array (n,) - kernel evaluations (float64)
        """
        # Ensure float64 precision
        z1 = np.asarray(z1, dtype=np.float64)
        z2 = np.asarray(z2, dtype=np.float64)
        lengthscales = np.asarray(lengthscales, dtype=np.float64)

        # Compute squared distances weighted by lengthscales
        # diff shape: (n, d)
        diff = z1[np.newaxis, :] - z2

        # Weighted squared distance: sum((diff / lengthscales)^2)
        weighted_sq_dist = np.sum((diff / lengthscales)**2, axis=1)

        # Squared exponential kernel
        return variance * np.exp(-0.5 * weighted_sq_dist)

    def _compute_sq_exp_kernel_diag(self, variance):
        """
        Compute diagonal (self) squared exponential kernel k(z, z).

        For squared exponential kernel, the self-covariance is always the output variance
        since the distance from a point to itself is 0.

        Args:
            variance: float - output variance

        Returns:
            float - kernel self-evaluation (always equals variance for SE kernel)
        """
        return variance

    def compute_gammas(self, zq):
        """
        Compute gamma matrices for SOCP filter using precomputed quantities.

        This follows the computation in ZeroMeanAffineGP.compute_gammas() but
        entirely in numpy for a single query point.

        Args:
            zq: numpy array (input_dim-2,) - query state (flat state, no control input)

        Returns:
            gamma_1: numpy array (1, 1) - constant mean term
            gamma_2: numpy array (2, 1) - linear mean coefficient (wrt u)
            gamma_3: numpy array (1, 1) - constant covariance term
            gamma_4: numpy array (2, 1) - linear covariance coefficient (wrt u)
            gamma_5: numpy array (2, 2) - quadratic covariance coefficient (wrt u)
        """
        # Compute cross-covariances between query and training data
        k_a = self._compute_sq_exp_kernel(zq, self.z_train,
                                          self.lengthscales_alpha,
                                          self.var_alpha)  # (n_train,)
        k_b1 = self._compute_sq_exp_kernel(zq, self.z_train,
                                           self.lengthscales_beta1,
                                           self.var_beta1)  # (n_train,)
        k_b2 = self._compute_sq_exp_kernel(zq, self.z_train,
                                           self.lengthscales_beta2,
                                           self.var_beta2)  # (n_train,)

        # Reshape for matrix operations
        k_a = k_a.reshape(1, -1)  # (1, n_train)
        k_b1 = k_b1.reshape(1, -1)  # (1, n_train)
        k_b2 = k_b2.reshape(1, -1)  # (1, n_train)

        # Stack beta kernels: k_beta shape (2, n_train)
        k_beta = np.vstack([k_b1, k_b2])

        # Multiply by u_train to get k_b: (2, n_train) * (n_train, 2).T element-wise
        # k_b[i, j] = k_beta[i, j] * u_train[j, i]
        k_b = k_beta * self.u_train.T  # (2, n_train)

        # Compute self-kernels for diagonal terms
        k_a_diag = self._compute_sq_exp_kernel_diag(self.var_alpha)
        k_b1_diag = self._compute_sq_exp_kernel_diag(self.var_beta1)
        k_b2_diag = self._compute_sq_exp_kernel_diag(self.var_beta2)

        # Compute gammas using precomputed K_inv_y and K_inv
        # gamma_1: (1, 1) - constant mean term
        gamma_1 = k_a @ self.K_inv_y  # (1, n_train) @ (n_train, 1) -> (1, 1)

        # gamma_2: (2, 1) - linear mean term
        gamma_2 = k_b @ self.K_inv_y  # (2, n_train) @ (n_train, 1) -> (2, 1)

        # gamma_3: (1, 1) - constant covariance term
        # k_a_diag - k_a @ K_inv @ k_a.T
        gamma_3 = k_a_diag - (k_a @ self.K_inv @ k_a.T)  # scalar -> (1, 1)
        gamma_3 = gamma_3.reshape(1, 1)

        # gamma_4: (2, 1) - linear covariance term
        # -2 * k_b @ K_inv @ k_a.T
        gamma_4 = -2 * (k_b @ self.K_inv @ k_a.T)  # (2, n_train) @ (n_train, n_train) @ (n_train, 1) -> (2, 1)

        # gamma_5: (2, 2) - quadratic covariance term
        # diag([k_b1_diag, k_b2_diag]) - k_b @ K_inv @ k_b.T
        k_beta_diag = np.array([[k_b1_diag, 0], [0, k_b2_diag]])
        gamma_5 = k_beta_diag - (k_b @ self.K_inv @ k_b.T)  # (2, 2)

        return gamma_1, gamma_2, gamma_3, gamma_4, gamma_5

    def predict_mean_cov(self, zq, uq):
        """
        Compute mean and covariance from gammas for a given state and control input.

        mean = gamma_1 + gamma_2.T @ u
        cov = gamma_3 + gamma_4.T @ u + u.T @ gamma_5 @ u + noise_var

        Args:
            zq: numpy array (input_dim-2,) - query state
            uq: numpy array (2,) - query control input

        Returns:
            mean: float - predicted mean
            cov: float - predicted variance (scalar)
        """
        gamma_1, gamma_2, gamma_3, gamma_4, gamma_5 = self.compute_gammas(zq)

        # Reshape uq for matrix operations
        u = uq.reshape(2, 1)  # (2, 1)

        # Compute mean
        mean = gamma_1 + gamma_2.T @ u  # (1, 1) + (1, 2) @ (2, 1) -> (1, 1)
        mean = mean.item()  # Extract scalar

        # Compute covariance
        cov = gamma_3 + gamma_4.T @ u + u.T @ gamma_5 @ u + self.noise_var
        cov = cov.item()  # Extract scalar

        return mean, cov


