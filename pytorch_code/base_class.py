import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """ Encoder (Variational Approximation to Posterior q(z|x, \phi) """
    def __init__(self, approx_params):
        super(Encoder, self).__init__()
        self.approx_params = approx_params

    def entropy(self, x):
        """ Return entropy of variational approx
            E_q[\log q(z | x, phi)]
        """
        raise NotImplementedError()

    def sample(self, x):
        """ Return a sample z from q(z | x, phi) """
        raise NotImplementedError()

class Decoder(nn.Module):
    """ Decoder Model to p(x, z | \theta) """
    def __init__(self, model_params):
        super(Decoder, self).__init__()
        self.model_params = model_params

    def generate_data(self, N):
        """ Generate Samples of X, Z """
        raise NotImplementedError()

    def loglike(self, x, z):
        """ Return loglikelihood of obs + latents
            $\log Pr(X, Z | \theta)$
        """
        raise NotImplementedError()

    def predict_x(self, z):
        raise NotImplementedError()

class VI(nn.Module):
    """ Stochastic Gradient VI """
    def __init__(self, encoder, decoder, **kwargs):
       super(VI, self).__init__()
       self.decoder = decoder
       self.encoder = encoder

    def elbo(self, x):
        z = self.encoder.sample(x=x)
        entropy = self.encoder.entropy(x=x)
        loglike = self.decoder.loglike(x=x, z=z)
        elbo_hat = loglike + entropy
        return elbo_hat/x.shape[1]

    def predict_x(self, x):
        z = self.encoder.sample(x=x)
        x_predict = self.decoder.predict_x(z)
        return x_predict



