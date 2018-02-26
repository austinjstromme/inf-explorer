# Decoder (Models)
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_class import Decoder

class LDSDecoder(Decoder):
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
        super(LDSDecoder, self).__init__(model_params=model_params)

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
        x = t.zeros((self.S, self.n))

        noise = t.distributions.Normal(t.zeros(self.m), self.LR)
        for s in xrange(0, self.S):
            eps = noise.sample()
            x[s] = t.matmul(self.C, z[s]) + noise

        return x

    def loglike(self, x, z):
        z = z.permute(1, 2, 0)
        x = x.permute(1, 2, 0)
        emit_dist = t.distributions.Normal(
                t.matmul(self.C, z[0]), self.LR)
        loglike = emit_dist.log_prob(x[0])
        for s in xrange(1, self.S):
            emit_dist = t.distributions.Normal(
                    t.matmul(self.C,z[s]), self.LR)
            trans_dist = t.distributions.Normal(
                    t.matmul(self.A, z[s-1]), self.LQ)

            p_emit = emit_dist.log_prob(x[s])
            p_tran = trans_dist.log_prob(z[s])

            loglike += p_emit + p_tran

        return t.mean(loglike)

    def generate_data(self, N):
        A = np.array(self.A.data.tolist())
        C = np.array(self.C.data.tolist())
        Q = np.array(self.Q.data.tolist())
        R = np.array(self.R.data.tolist())

        Zs = [None] * N
        Xs = [None] * N

        for n in range(N):
            z = np.zeros((self.S, self.m))
            x = np.zeros((self.S, self.n))
            z_prev = np.random.multivariate_normal(
                    mean = np.zeros(self.m),
                    cov = Q*2,
                    )
            for s in range(0,self.S):
                z_cur = np.random.multivariate_normal(
                        mean=np.dot(A, z_prev),
                        cov=Q,
                        )
                x_cur = np.random.multivariate_normal(
                        mean=np.dot(C, z_cur),
                        cov=R,
                        )

                z[s] = z_cur
                x[s] = x_cur
                z_prev = z_cur

            Zs[n] = z
            Xs[n] = x

        Z = np.stack(Zs, axis=0)
        X = np.stack(Xs, axis=0)

        return X, Z


