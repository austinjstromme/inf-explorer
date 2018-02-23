# Decoder (Models)
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_class import Decoder

class SeqDecoder(Decoder):
    """
    A map from hidden states in \R^m to means in \R^n.

    Attributes:
        m: dimension of hidden state space
        n: dimension of observation space
        S: length of series
        C: matrix in \R^m \times \R^n (variable)
        R: matrix in \R^m \times \R^m governing variance of observations
        A: matrix in \R^n \times \R^n (variable)
        Q: matrix in \R^n \times \R^n latent state innovation variance
    """
    def __init__(self, **model_params):
        super(SeqDecoder, self).__init__(model_params=model_params)

        self.m = model_params['m']
        self.n = model_params['n']
        self.S = model_params['S']

        self.A = nn.Parameter(t.from_numpy(model_params['A']).float())
        self.LQ = nn.Parameter(t.from_numpy(
            np.linalg.cholesky(model_params['Q'])
            ).float())
        self.Q = t.matmul(self.LQ, t.t(self.LQ))
        self.C = nn.Parameter(t.from_numpy(model_params['C']).float())
        self.LR = nn.Parameter(t.from_numpy(
            np.linalg.cholesky(model_params['R'])
            ).float())
        self.R = t.matmul(self.LR, t.t(self.LR))


    def predict_x(self, z):
        if (z.size() != self.m*self.S):
            raise ValueError("z is of wrong size")

        x = t.zeros((self.S, self.n))

        noise = t.distributions.Normal(t.zeros(self.m), self.R)
        for s in xrange(0, self.S):
            eps = noise.sample()
            x[s] = t.mv(self.C, z[s]) + noise

        return x

    def loglike(self, x, z):
#        if (np.prod(x.size()) != self.n*self.S):
#            raise ValueError("x is of wrong size")
#        if (np.prod(z.size()) != self.m*self.S): raise ValueError("z is of wrong size")
#
        emit_dist = t.distributions.Normal(t.mv(self.C, z[0]), self.R)
        res = emit_dist.log_prob(x[0])
        for s in xrange(1, self.S):
            emit_dist = t.distributions.Normal(t.mv(self.C, z[s]), self.R)
            trans_dist = t.distributions.Normal(t.mv(self.A, z[s - 1]), self.Q)

            p_emit = emit_dist.log_prob(x[s])
            p_tran = trans_dist.log_prob(z[s])

            res += p_emit + p_tran

        return res

    def generate_data(self):
        # TODO: implement
        raise NotImplemented()
