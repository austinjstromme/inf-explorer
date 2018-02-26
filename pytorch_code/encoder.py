# Encoder (Variational Approximation)
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base_class import Encoder

class MeanFieldGaussian(Encoder):
    """ Mean Field Approximation
        q(z | x) = \prod N(z_i | mu(x_i), sigma(x_i)**2)


        Args:
            nn_mu (nn): takes x_i returns mu_i
            nn_log_lambduh (nn): takes x_i returns log_lambda_i (precision)
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
        return -0.5*t.sum(t.mean(log_lambduh, dim=0))

    def sample(self, x):
        """ Return a sample z from q(z | x, phi) """
        mu, log_lambduh = self.nn_mu(x), self.nn_log_lambduh(x)
        sigma = t.exp(-0.5*log_lambduh)
        return mu + sigma * Variable(t.randn(sigma.size()), requires_grad=False)

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



