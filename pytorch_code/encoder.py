# Encoder (Variational Approximation)
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base_class import Encoder

class MeanFieldGaussian(Encoder):
    """ Mean Field Approximation
        q(z | x) = \prod N(z_i | mu(x_i), sigma(x_i)**2)

        Args:
            nn_mu (nn): takes x returns mu
            nn_log_lambduh (nn): takes x returns log_lambda (precision)
                lambda_i = sigma**-2
    """
    def __init__(self, nn_mu, nn_log_lambduh, **approx_params):
        super(MeanFieldGaussian, self).__init__(approx_params=approx_params)
        self.nn_mu = nn_mu
        self.nn_log_lambduh = nn_log_lambduh

    def entropy(self, x):
        """ Return entropy of variational approx
            E_q[\log q(z | x, phi)]
        """
        log_lambduh = self.nn_log_lambduh(x)
        return -0.5*t.sum(t.mean(log_lambduh, dim=0)) + \
                0.5*x.shape[2]*np.log(2.0*np.pi*np.exp(1))

    def sample(self, x):
        """ Return a sample z from q(z | x, phi) """
        mu, log_lambduh = self.nn_mu(x), self.nn_log_lambduh(x)
        sigma = t.exp(-0.5*log_lambduh)
        return mu + sigma * Variable(t.randn(sigma.size()), requires_grad=False)

    def get_mu_sigma(self, x):
        """ Return mean and diagonal variance of z given x"""
        mu, log_lambduh = self.nn_mu(x), self.nn_log_lambduh(x)
        sigma = t.exp(-0.5*log_lambduh)
        return mu.data, sigma.data

class TriDiagInvGaussian(Encoder):
    """ Tridiagonal Precision Gaussian

        q(z | x) = Normal( z | mu, Sigma), where Sigma**-1 is tridiagonal


    Args:
        nn_mu (nn): takes x returns mu
        nn_log_alpha (nn): takes x return log_alpha
        nn_beta (nn): takes x return beta

    Notes:
        Sigma**-1 = L L^T where diag(L) = alpha, diag(L, -1) = beta
                L is the cholesky decomposition of Sigma
        If x is N by T by 1, then
            mu is N by T by 1
            alpha is N by T by 1
            beta is N by T by 1 (but beta[T-1] is not used)

    """
    def __init__(self, nn_mu, nn_log_alpha, nn_beta, **approx_params):
        super(TriDiagInvGaussian, self).__init__(approx_params=approx_params)
        self.nn_mu = nn_mu
        self.nn_log_alpha = nn_log_alpha
        self.nn_beta = nn_beta

    def entropy(self, x):
        """ Return entropy of variational approx
            E_q[\log q(z | x, phi)] = -0.5 * det(Sigma**-1)

        """
        log_alpha = self.nn_log_alpha(x)
        alpha = t.exp(log_alpha)
        beta = self.nn_beta(x)

        # Recursive computation for tridiagonal
        n = x.shape[0] # batch size
        T = x.shape[1] # time points
        logf_prevprev = Variable(t.zeros(x.shape[0]), requires_grad=False)
        logf_prev = 2.0*log_alpha[:,0,0]
        logdet = logf_prev
        for ii in range(1, T):
            a = alpha[:,ii,0]**2 + beta[:,ii-1,0]**2
            b = alpha[:,ii-1,0] * beta[:,ii-1,0]
            logdet = t.log(a - b**2 * t.exp(logf_prevprev-logf_prev)) \
                    + logf_prev

            if (logdet != logdet).any():
                raise RuntimeError()

            # recurse
            logf_prevprev, logf_prev = logf_prev, logdet

        return -0.5*t.sum(logdet/T, dim=0) + \
                0.5*x.shape[2]*np.log(2.0*np.pi*np.exp(1))

    def _get_cholesky_precision_matrix_diags(self, x):
        # Return the Diagonal + Off Diagonal of L(Sigma**-1)
        alpha = np.array(t.exp(self.nn_log_alpha(x)).data.tolist())[:,:,0]
        beta = np.array(self.nn_beta(x).data.tolist())[:,:,0]
        N = x.shape[0] # batch size
        T = x.shape[1] # time points
        a = np.zeros([N,T])
        b = np.zeros([N,T])
        a[:,0] = alpha[:,0]**2
        for ii in range(1, T):
            a[:,ii] = alpha[:,ii]**2 + beta[:,ii-1]**2
            b[:,ii-1] = alpha[:,ii-1] * beta[:,ii-1]
        return a, b

    def sample(self, x):
        """ Return a sample z from q(z | x, phi) """
        T = x.shape[1]
        mu = self.nn_mu(x)
        alpha = t.exp(self.nn_log_alpha(x))
        beta = self.nn_beta(x)
        white_noise = Variable(t.randn(mu.size()), requires_grad=False)

        z_sample = [None for _ in range(T)]
        # Use Tridiagonal matrix algorithm to get noise covariance
        z_sample[T-1] = white_noise[:, T-1] / alpha[:, T-1]
        for ii in reversed(range(0, T-1)):
            z_sample[ii] = (white_noise[:,ii] - \
                    beta[:,ii] * z_sample[ii+1])/ alpha[:, ii]
        z_sampled = t.transpose(t.stack(z_sample), 0, 1) + mu
        return z_sampled

    def get_mu_sigma(self, x):
        """ Return mean and diagonal variance of z given x"""
        mu = self.nn_mu(x).data
        alpha = t.exp(self.nn_log_alpha(x)).data
        beta = self.nn_beta(x).data
        sigma = t.zeros_like(mu)

        # Recursion for calcuating Sigma
        a = t.zeros_like(mu)
        b = t.zeros_like(mu)
        a[:, 0] = alpha[:, 0]**2
        for ii in range(alpha.shape[1]-1):
            b[:,ii] = alpha[:, ii]*beta[:, ii]
            a[:,ii+1] = alpha[:, ii+1]**2 + beta[:, ii]**2

        log_theta = t.zeros_like(mu)
        log_theta[:, 0] = t.log(a[:, 0])
        log_phi = t.zeros_like(mu)
        log_phi[:, -1] = t.log(a[:, -1])

        if alpha.shape[1] > 1:
            log_theta[:, 1] = log_theta[:, 0] + t.log(a[:, 1] - b[:, 0]**2*t.exp(-log_theta[:,0]))
            for ii in range(2, alpha.shape[1]):
                log_theta[:, ii] = log_theta[:, ii-1] + t.log(a[:, ii]  - b[:, ii-1]**2 * t.exp(log_theta[:, ii-2] - log_theta[:,ii-1]))
            log_phi[:, -2] = log_phi[:,-1] + t.log(a[:, -2] - b[:, -2]**2*t.exp(-log_phi[:,-1]))
            for ii in reversed(range(0, alpha.shape[1]-2)):
                log_phi[:, ii] = log_phi[:, ii+1] + t.log(a[:, ii] - b[:, ii]**2 * t.exp(log_phi[:, ii+2] - log_phi[:, ii+1]))

        if alpha.shape[1] == 1:
            sigma[:, 0] = t.exp(0.5*log_theta[:, 0])
        else:
            sigma[:, 0] = t.exp(0.5*(log_phi[:, 1] - log_theta[:,-1]))
            sigma[:, -1] = t.exp(0.5*(log_theta[:, -2] - log_theta[:,-1]))
            for ii in range(1, alpha.shape[1]-1):
                sigma[:, ii] = t.exp(0.5*(log_theta[:,ii-1] + log_phi[:,ii+1] - log_theta[:,-1]))

        return mu, sigma


# Helper Functions
def elementwise_NN(input_dim, latent_dim,
        h_dims=[10,10], layerNN=nn.SELU):
    """ Return elementwise nn """
    layer_dims = [input_dim]+h_dims+[latent_dim]
    NN_layers = [x for y in [
            [nn.Linear(layer_dims[ii], layer_dims[ii+1]), layerNN()]
            for ii in range(len(layer_dims)-1)
            ] for x in y ]
    NN_layers = NN_layers[:-1] # Drop last nonlinear activation
    NN = nn.Sequential(*NN_layers)
    return NN

def dial_conv_NN(input_dim, latent_dim,
        h_dims = [10,10], kernel_sizes = [3, 3], dilations = [1, 2],
        layerNN=nn.SELU):
    kernel_sizes = kernel_sizes + [1]
    dilations = dilations + [1]
    padding_sizes = [(kernel_size-1)*dilation/2
            for kernel_size, dilation in zip(kernel_sizes, dilations)]
    layer_dims = [input_dim]+h_dims+[latent_dim]

    if layerNN is not None:
        NN_conv_layers = [x
                for y in [
                    [nn.Conv1d(layer_dims[ii], layer_dims[ii+1],
                        kernel_size=kernel_sizes[ii],
                        padding=padding_sizes[ii],
                        dilation=dilations[ii]
                        ),
                    layerNN()]
                    for ii in range(len(layer_dims)-1)]
                for x in y]
        NN_conv_layers = NN_conv_layers[:-1] # Drop last layer
    else:
        NN_conv_layers = [
                nn.Conv1d(layer_dims[ii], layer_dims[ii+1],
                        kernel_size=kernel_sizes[ii],
                        padding=padding_sizes[ii],
                        dilation=dilations[ii]
                        )
                for ii in range(len(layer_dims)-1)
                ]

    NN = ConvWrap(nn=nn.Sequential(*NN_conv_layers))
    return NN

def elementwise_MFGaussian(input_dim, latent_dim,
        h_dims=[10,10], layerNN=nn.SELU):
    """ Return elementwise nn_mu and nn_log_lambduh nn.Modules """
    layer_dims = [input_dim]+h_dims+[latent_dim]
    mu_layers = [x for y in [
            [nn.Linear(layer_dims[ii], layer_dims[ii+1]), layerNN()]
            for ii in range(len(layer_dims)-1)
            ] for x in y ]
    mu_layers = mu_layers[:-1] # Drop last layer
    lambduh_layers = [x for y in [
            [nn.Linear(layer_dims[ii], layer_dims[ii+1]), layerNN()]
            for ii in range(len(layer_dims)-1)
            ] for x in y ]
    lambduh_layers = lambduh_layers[:-1] # Drop last layer
    nn_mu = nn.Sequential(*mu_layers)
    nn_log_lambduh = nn.Sequential(*lambduh_layers)
    return nn_mu, nn_log_lambduh

def dial_conv_MFGaussian(input_dim, latent_dim,
        h_dims = [10,10], kernel_sizes = [3, 3], dilations = [1, 2],
        layerNN=nn.SELU):

    kernel_sizes = kernel_sizes + [1]
    dilations = dilations + [1]
    padding_sizes = [(kernel_size-1)*dilation/2
            for kernel_size, dilation in zip(kernel_sizes, dilations)]
    layer_dims = [input_dim]+h_dims+[latent_dim]

    if layerNN is not None:
        mu_conv_layers = [x
                for y in [
                    [nn.Conv1d(layer_dims[ii], layer_dims[ii+1],
                        kernel_size=kernel_sizes[ii],
                        padding=padding_sizes[ii],
                        dilation=dilations[ii]
                        ),
                    layerNN()]
                    for ii in range(len(layer_dims)-1)]
                for x in y]
        mu_conv_layers = mu_conv_layers[:-1] # Drop last layer
        lambduh_conv_layers = [x
                for y in [
                    [nn.Conv1d(layer_dims[ii], layer_dims[ii+1],
                        kernel_size=kernel_sizes[ii],
                        padding=padding_sizes[ii],
                        dilation=dilations[ii]
                        ),
                    layerNN()]
                    for ii in range(len(layer_dims)-1)]
                for x in y]
        lambduh_conv_layers = lambduh_conv_layers[:-1] # Drop last layer
    else:
        mu_conv_layers = [
                nn.Conv1d(layer_dims[ii], layer_dims[ii+1],
                        kernel_size=kernel_sizes[ii],
                        padding=padding_sizes[ii],
                        dilation=dilations[ii]
                        )
                for ii in range(len(layer_dims)-1)
                ]
        lambduh_conv_layers = [
                nn.Conv1d(layer_dims[ii], layer_dims[ii+1],
                        kernel_size=kernel_sizes[ii],
                        padding=padding_sizes[ii],
                        dilation=dilations[ii]
                        )
                for ii in range(len(layer_dims)-1)
                ]

    nn_mu = ConvWrap(nn=nn.Sequential(*mu_conv_layers))
    nn_log_lambduh = ConvWrap(nn=nn.Sequential(*lambduh_conv_layers))

    return nn_mu, nn_log_lambduh

class ConvWrap(nn.Module):
    def __init__(self, nn):
        super(ConvWrap, self).__init__()
        self.nn = nn

    def forward(self, x):
        # x is batch by time by dim
        x = x.permute(0, 2, 1)
        # x is batch by dim by time
        z_out = self.nn(x)
        # swap channel + time back
        z_out = z_out.permute(0, 2, 1)
        return z_out

# Example

if __name__ == "__main__":

    input_dim = 10 # Dimension of x_s
    latent_dim = 2 # Dimension of z_s

    nn_mu, nn_log_lambduh = elementwise_MFGaussian(input_dim, latent_dim)
    encoder = MeanFieldGaussian(nn_mu, nn_log_lambduh)



