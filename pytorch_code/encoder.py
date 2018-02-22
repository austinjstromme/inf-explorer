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
        log_lambduh = self.nn_log_lambda(x)
        return -0.5*t.mean(lambduh)

    def sample(self, x):
        """ Return a sample z from q(z | x, phi) """
        mu, log_lambduh = self.nn_mu(x), self.nn_log_lambduh(x)
        sigma = t.exp(-1.0*log_lambduh/2.0)
        return mu + sigma * Variable(t.randn(sigma.size()), requires_grad=False)

def elementwise_MFGaussian(input_dim, latent_dim,
        h_dims=[100,100], layerNN=nn.ReLU):
    """ Return elementwise nn_mu and nn_log_lambduh nn.Modules """
    layer_dims = [input_dim]+h_dims+[latent_dim]
    mu_layers = [
            layerNN(nn.Linear(layer_dims[ii], layer_dims[ii+1]))
            for ii in range(len(layer_dims)-1)
            ]
    lambduh_layers = [
            layerNN(nn.Linear(layer_dims[ii], layer_dims[ii+1]))
            for ii in range(len(layer_dims)-1)
            ]
    nn_mu = nn.Sequential(*mu_layers)
    nn_log_lambduh = nn.Sequential(*lambduh_layers)
    nn_mu = ElementwiseNNWrap(nn_mu)
    nn_log_lambduh = ElementwiseNNWrap(nn_log_lambduh)
    return nn_mu, nn_log_lambduh

#def dial_conv_MFGaussian(input_dim, latent_dim,
#        kernel_sizes = [3, 3], dilations = [1, 2]):
#
#    in_dim = input_dim
#    for kernel_size, dilation in zip(kernel_sizes, dilations):
#        conv = nn.Conv1d(



# Helper Classes
class ElementwiseNNWrap(nn.Module):
    """ Wraps input nn.Module to be applied elementwise over dim """
    def __init__(self, nn, dim=0):
        super(ElementwiseNNWrap, self).__init__()
        self.nn = nn
        self.dim = dim

    def forward(self, x):
        return t.concat([self.nn(x_i) for x_i in x.split(self.dim)])

class WindowNNWrap(nn.Module):
    """ Wraps input nn.Module to be applied over sliding window dim """
    def __init__(self, nn, dim=0, window=1):
        super(WindowNNWrap, self).__init__()
        self.nn = nn
        self.dim = dim

    def forward(self, x):
        raise NotImplementedError()

# Example

if __name__ == "__main__":

    input_dim = 10 # Dimension of x_s
    latent_dim = 2 # Dimension of z_s

    nn_mu, nn_log_lambduh = elementwise_MFGaussian(input_dim, latent_dim)
    encoder = MeanFieldGaussian(nn_mu, nn_log_lambduh)



