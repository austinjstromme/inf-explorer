#
import numpy as np
from base_class import VI
from encoder import MeanFieldGaussian, elementwise_MFGaussian
from decoder import LDSDecoder
from data_generator import generate_lds_data
import torch as t
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from tqdm import tqdm

n = 1 # Dimension of x_s
m = 1 # Dimension of z_s
batch_size = 16 # Batchsize
N = 64 # Number of replications
S = 100 # Length of Timeseries


# LDS Model (really mf gaussian since A = 0)
A = np.eye(m)*0.0
Q = np.eye(m)*1.0
C = 2.0*np.block([[np.eye(m,m)], [np.random.normal(size=(n-m, m))]])
R = np.eye(n)*0.01

decoder = LDSDecoder(m=m, n=n, S=S, C=C, R=R, A=A, Q=Q)
X, Z = decoder.generate_data(N=N)

# VI approx
nn_mu, nn_log_lambduh = elementwise_MFGaussian(input_dim=n, latent_dim=m,
        h_dims=[])
encoder = MeanFieldGaussian(nn_mu, nn_log_lambduh)


# Combine to Do VI
vi = VI(encoder=encoder, decoder=decoder)


# Only train encoder
optimizer = t.optim.Adam(vi.encoder.parameters(), lr=0.1)

varX = Variable(t.from_numpy(X).float())
varZ = Variable(t.from_numpy(Z).float())

dataset = TensorDataset(t.from_numpy(X).float(), t.from_numpy(Z).float())
dataloader = t.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=True)

p_bar = tqdm(range(100))
for epoch in p_bar:
    elbo_hat, counts = 0.0, 0.0
    for i, data in enumerate(dataloader, 0):
        observations, latent_vars = data
        observations = Variable(observations.resize_(batch_size, S, n))
        optimizer.zero_grad()
        loss = -1.0*vi.elbo(observations)
        loss.backward(retain_graph=True)
        optimizer.step()
        elbo_hat = (elbo_hat*counts-1.0*loss.data[0])/(counts+1)
        counts += 1
        p_bar.set_description("Epoch {0}, ELBO {1}".format(epoch, elbo_hat))
    print("")

    if epoch % 10 == 0:
        batch_index = 10
        mu_z = np.array(vi.encoder.nn_mu(varX[batch_index]).data.tolist())
        sigma_z = np.exp(-0.5*np.array(vi.encoder.nn_log_lambduh(varX[batch_index]).data.tolist()))
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(X[batch_index], "o", label="X")
        plt.plot(Z[batch_index], "-", label="Z")
        plt.plot(mu_z, "--k", label="Eq(Z|X)")
        plt.plot(mu_z+sigma_z, ":k", label="STDq(Z|X)")
        plt.plot(mu_z-sigma_z, ":k", label=None)
        plt.title("Plot of data, true latent, and variational mean\n Epoch={0} Seq={1}".format(epoch, batch_index))
        plt.legend()
        plt.show()


