import matplotlib.pyplot as plt
import numpy as np
import torch as t
from torch.autograd import Variable

def plot_encoder(encoder, x, z,
        num_post_draws=20, ax=None):
    var_x = Variable(t.from_numpy(np.array([x])).float())
    mu_z, sigma_z = encoder.get_mu_sigma(var_x)
    mu_z, sigma_z = np.array(mu_z.tolist())[0], np.array(sigma_z.tolist())[0]

    z_samples = [np.array(encoder.sample(var_x).data[0].tolist())
            for _ in range(num_post_draws)]

    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax.plot(x, "o", label="X")
    ax.plot(z, "-", label="Z")

    ax.plot(mu_z, "--k", label="Eq(Z|X)")
    ax.plot(mu_z+sigma_z, ":k", label="STDq(Z|X)")
    ax.plot(mu_z-sigma_z, ":k", label=None)
    for ii, z_sample in enumerate(z_samples):
        if ii == 0:
            ax.plot(z_sample, "-", color='gray', alpha=0.25,
                    label="Z* from q")
        else:
            ax.plot(z_sample, "-", color='gray', alpha=0.25)
    ax.legend()
    return ax

def plot_encoder_resid(encoder, x, z,
        num_post_draws=20, ax=None):
    var_x = Variable(t.from_numpy(np.array([x])).float())
    mu_z, sigma_z = encoder.get_mu_sigma(var_x)
    mu_z, sigma_z = np.array(mu_z.tolist())[0], np.array(sigma_z.tolist())[0]

    z_samples = [np.array(encoder.sample(var_x).data[0].tolist())
            for _ in range(num_post_draws)]

    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax.plot(np.zeros_like(x), "-", label="Z-Z")
    ax.plot(mu_z-z, "--k", label="Eq(Z|X)-Z")
    ax.plot(mu_z-z+sigma_z, ":k", label="STDq(Z|X)-Z")
    ax.plot(mu_z-z-sigma_z, ":k", label=None)
    for ii, z_sample in enumerate(z_samples):
        if ii == 0:
            ax.plot(z_sample-z, "-", color='gray', alpha=0.25,
                    label="Z* - Z")
        else:
            ax.plot(z_sample-z, "-", color='gray', alpha=0.25)
    ax.legend()
    return ax

def plot_latent_state(w, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    t = np.arange(0, len(w))
    for k in range(0,int(np.max(w)+1)):
        ax.plot(t[w==k], np.ones(np.sum(w==k))*k, "o", color="C{0}".format(k))
    return ax


