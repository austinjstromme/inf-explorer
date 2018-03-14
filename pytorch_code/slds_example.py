#
import numpy as np
import matplotlib.pyplot as plt
from base_class import VI
from encoder import (
        MeanFieldGaussian, elementwise_MFGaussian, dial_conv_MFGaussian,
        TriDiagInvGaussian, elementwise_NN, dial_conv_NN,
        )
from decoder import SLDSDecoder
from data_generator import generate_lds_data
import torch as t
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from tqdm import tqdm

np.random.seed(1234)
t.manual_seed(1234)

n = 1 # Dimension of x_s
m = 1 # Dimension of z_s
batch_size = 16 # Batchsize
N = 32 # Number of replications
S = 100 # Length of Timeseries


# SLDS Model
print("Setting Up SLDS Model Decoder")
Pi = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]])
A = np.array([np.eye(m)*0.99, np.eye(m)*0.99, np.eye(m)*0.99])
Delta_Q = np.array([np.ones(m), np.zeros(m), -np.ones(m)])*0.05
Q = np.array([np.eye(m), np.eye(m), np.eye(m)])*0.1
C = np.array([np.eye(n), np.eye(n), np.eye(n)])
R = np.array([np.eye(n), np.eye(n), np.eye(n)])*1.0
num_states = np.shape(A)[0]

decoder = SLDSDecoder(num_states=num_states, m=m, n=n, S=S, C=C, R=R, A=A, Q=Q,
        Delta_Q=Delta_Q, Pi=Pi)
print("Generating Data")
X, Z, W = decoder.generate_data(N=N)
varX = Variable(t.from_numpy(X).float())
varZ = Variable(t.from_numpy(Z).float())

# VI approx
print("Setting Up VI Encoder")
#nn_mu, nn_log_lambduh = elementwise_MFGaussian(input_dim=n, latent_dim=m,
#        h_dims=[])
nn_mu, nn_log_lambduh = dial_conv_MFGaussian(input_dim=n, latent_dim=m,
        h_dims=[10, 10, 10], kernel_sizes = [3, 3, 3], dilations=[1,2,4],
        layerNN=t.nn.SELU)

encoder = MeanFieldGaussian(nn_mu, nn_log_lambduh)


nn_mu = dial_conv_NN(input_dim=n, latent_dim=m, h_dims=[10,10,10],
        kernel_sizes=[3, 3, 3], dilations=[1,2,4], layerNN=t.nn.SELU)
nn_log_alpha = dial_conv_NN(input_dim=n, latent_dim=m, h_dims=[10,10,10],
        kernel_sizes=[3, 3, 3], dilations=[1,2,4], layerNN=t.nn.SELU)
nn_beta = dial_conv_NN(input_dim=n, latent_dim=m, h_dims=[10,10,10],
        kernel_sizes=[3, 3, 3], dilations=[1,2,4], layerNN=t.nn.SELU)
encoder = TriDiagInvGaussian(nn_mu, nn_log_alpha, nn_beta)

# Combine to Do VI
print("Setting up optimization problem")
vi = VI(encoder=encoder, decoder=decoder)

# Only train encoder
optimizer = t.optim.Adam(vi.encoder.parameters(), lr=0.01)


dataset = TensorDataset(t.from_numpy(X).float(), t.from_numpy(Z).float())
dataloader = t.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=True)

p_bar = tqdm(range(11))
for epoch in p_bar:
    elbo_hat, counts = 0.0, 0.0
    for i, data in enumerate(dataloader, 0):
        observations, latent_vars = data
        observations = Variable(observations.resize_(batch_size, S, n))
        optimizer.zero_grad()
        loss = -1.0*vi.elbo(observations)
        if np.isnan(loss.data[0]):
            raise RuntimeError()
        loss.backward(retain_graph=True)
        optimizer.step()
        elbo_hat = (elbo_hat*counts-1.0*loss.data[0])/(counts+1)
        counts += 1
        p_bar.set_description("Epoch {0}, ELBO {1}".format(epoch, elbo_hat))
    print("")

    if epoch % 10 == 0:
        batch_index = 10
##        mu_z = np.array(vi.encoder.nn_mu(varX[batch_index:batch_index+1]).data.tolist())[0]
##        sigma_z = np.exp(-0.5*np.array(vi.encoder.nn_log_lambduh(varX[batch_index:batch_index+1]).data.tolist()))[0]
#        mu_z, sigma_z = vi.encoder.get_mu_sigma(varX[batch_index:batch_index+1])
#        mu_z, sigma_z = np.array(mu_z.tolist())[0], np.array(sigma_z.tolist())[0]
#        import matplotlib.pyplot as plt
#        plt.figure()
#        plt.plot(X[batch_index], "o", label="X")
#        plt.plot(Z[batch_index], "-", label="Z")
#        plt.plot(mu_z, "--k", label="Eq(Z|X)")
#        plt.plot(mu_z+sigma_z, ":k", label="STDq(Z|X)")
#        plt.plot(mu_z-sigma_z, ":k", label=None)
#        plt.title("Plot of data, true latent, and variational mean\n Epoch={0} Seq={1}".format(epoch, batch_index))
#        plt.legend()
#        plt.show()

        from figure_utils import (
                plot_encoder, plot_encoder_resid, plot_latent_state,
                )
        fig, axes = plt.subplots(2, 2, sharex=True, gridspec_kw={'height_ratios':[5,1]})
        plot_encoder(vi.encoder, X[batch_index], Z[batch_index], ax=axes[0,0])
        axes[0,0].set_title("Plot of data, true latent, and variational mean\n Epoch={0} Seq={1}".format(epoch, batch_index))
        plot_latent_state(W[batch_index], ax=axes[1,0])
        axes[1,0].set_ylabel("Discrete Latent State")
        plot_encoder_resid(vi.encoder, X[batch_index], Z[batch_index], ax=axes[0,1])
        axes[0,1].set_title("Plot of residuals, and variational approximation residuals\n Epoch={0} Seq={1}".format(epoch, batch_index))
        plot_latent_state(W[batch_index], ax=axes[1,1])
        plt.show()


