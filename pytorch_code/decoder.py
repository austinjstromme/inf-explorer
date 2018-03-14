# Decoder (Models)
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_class import Decoder
from torch.autograd import Variable

class LDSDecoder(Decoder):
    """
    Linear Gaussian State Space Model

    Attributes:
        m: dimension of hidden state space
        n: dimension of observation space
        S: length of series
        C: matrix in \R^m \times \R^n observation matrix
        R: matrix in \R^m \times \R^m governing variance of observations
        A: matrix in \R^n \times \R^n latent statne transition matrix
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
        x = t.zeros((z.shape[0], z.shape[1], self.n))

        noise = t.distributions.Normal(t.zeros(self.m), self.LR)
        for b in xrange(0, z.shape[0]):
            for s in xrange(0, z.shape[1]):
                eps = noise.sample()
                x[b,s] = t.matmul(self.C, z[b, s]) + noise

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

class SLDSDecoder(Decoder):
    """
    Switching Linear Gaussian State Space Model

    Attributes:
        num_states: number of latent states
        m: dimension of hidden state space
        n: dimension of observation space
        S: length of series

        Pi: matrix in num_states by num_states discrete latent state transition matrix
        A: matrix in num_states by n by n latent state transition matrix
        Delta_Q: vector in num_states by n latent state innovation drift
        Q: matrix in num_states by n by n latent state innovation variance
        C: matrix in num_states by m by n observation matrix
        R: matrix in num_states by m by m governing variance of observations
    """
    def __init__(self, **model_params):
        super(SLDSDecoder, self).__init__(model_params=model_params)

        self.num_states = model_params['num_states']
        self.m = model_params['m']
        self.n = model_params['n']
        self.S = model_params['S']
        if self.m != 1 or self.n != 1:
            raise NotImplementedError("n and m must be 1 (scalar x and z)")

        self.Pi = nn.Parameter(t.from_numpy(model_params['Pi']).float())
        self.A = nn.Parameter(t.from_numpy(model_params['A']).float())
        self.Delta_Q = nn.Parameter(t.from_numpy(model_params['Delta_Q']).float())
        self.Q = nn.Parameter(t.from_numpy(model_params['Q']).float())
        self.C = nn.Parameter(t.from_numpy(model_params['C']).float())
        self.R = nn.Parameter(t.from_numpy(model_params['R']).float())

    def predict_x(self, z):
        raise NotImplementedError()

    def _forward_pass_batched(self, x, z):
        z = z.permute(1, 2, 0)
        x = x.permute(1, 2, 0)
        prob_vector = Variable(
                t.ones((self.num_states, x.shape[-1]))/self.num_states
                )
        log_constant = Variable(t.zeros(x.shape[-1]))
        z_prev = Variable(t.zeros((self.n, x.shape[-1])), requires_grad=False)
        for s in range(0, x.shape[0]):
            z_cur = z[s]
            x_cur = x[s]

            # Log Pr(Y, X | X_prev)
            logP_s = Variable(t.zeros((self.num_states, x.shape[-1])))
            for k in range(self.num_states):
                emit_dist = t.distributions.Normal(
                        t.matmul(self.C[k], z_cur), t.sqrt(self.R[k]))
                trans_dist = t.distributions.Normal(
                        t.matmul(self.A[k], z_prev)+self.Delta_Q[k],
                        t.sqrt(self.Q[k]))
                logP_s[k] = emit_dist.log_prob(x_cur) + trans_dist.log_prob(z_cur)
            log_constant = log_constant + t.max(logP_s, dim=0)[0]
            P_s = t.exp(logP_s - t.max(logP_s, dim=0)[0])
            prob_vector = t.transpose(
                    t.matmul(t.transpose(prob_vector, 0,1), self.Pi), 0, 1,
                    )
            prob_vector = prob_vector * P_s
            log_constant = log_constant + t.log(t.sum(prob_vector, dim=0))
            prob_vector = prob_vector/t.sum(prob_vector, dim=0)

            z_prev = z_cur
        return log_constant

    def loglike(self, x, z):
        loglike = self._forward_pass_batched(x, z)
        return t.mean(loglike)

    def generate_data(self, N):
        Pi = np.array(self.Pi.data.tolist())
        A = np.array(self.A.data.tolist())
        Delta_Q = np.array(self.Delta_Q.data.tolist())
        Q = np.array(self.Q.data.tolist())
        C = np.array(self.C.data.tolist())
        R = np.array(self.R.data.tolist())

        Ws = [None] * N
        Zs = [None] * N
        Xs = [None] * N

        for n in range(N):
            w = np.zeros((self.S))
            z = np.zeros((self.S, self.m))
            x = np.zeros((self.S, self.n))
            w_prev = random_categorical(
                    np.ones(self.num_states)/self.num_states,
                    )
            z_prev = np.random.multivariate_normal(
                    mean = np.zeros(self.m),
                    cov = Q[w_prev]*2,
                    )
            for s in range(0,self.S):
                w_cur = random_categorical(Pi[w_prev])
                z_cur = np.random.multivariate_normal(
                        mean=np.dot(A[w_cur], z_prev) + Delta_Q[w_cur],
                        cov=Q[w_cur],
                        )
                x_cur = np.random.multivariate_normal(
                        mean=np.dot(C[w_cur], z_cur),
                        cov=R[w_cur],
                        )

                w[s] = w_cur
                z[s] = z_cur
                x[s] = x_cur
                w_prev = w_cur
                z_prev = z_cur

            Ws[n] = w
            Zs[n] = z
            Xs[n] = x

        W = np.stack(Ws, axis=0)
        Z = np.stack(Zs, axis=0)
        X = np.stack(Xs, axis=0)

        return X, Z, W

def random_categorical(pvals, size=None):
    out = np.random.multinomial(n=1, pvals=pvals, size=size).dot(
            np.arange(len(pvals)))
    return int(out)


