import numpy as np
from base_class import VI
from encoder import MeanFieldGaussian, elementwise_MFGaussian
from decoder import SeqDecoder
from data_generator import generate_lds_data
import torch as t
from torch.autograd import Variable
from tqdm import tqdm

n = 1 # Dimension of x_s
m = 1 # Dimension of z_s
S = 100 # Length of Timeseries


# LDS Model
A = np.eye(m)*0.9
Q = np.eye(m)*0.1
C = np.block([[np.eye(m,m)], [np.random.normal(size=(n-m, m))]])
R = np.eye(n)

data = generate_lds_data(T=S, A=A, Q=Q, C=C, R=R)


decoder = SeqDecoder(m=m, n=n, S=S, C=C, R=R, A=A, Q=Q)


# VI approx
nn_mu, nn_log_lambduh = elementwise_MFGaussian(input_dim=n, latent_dim=m)
encoder = MeanFieldGaussian(nn_mu, nn_log_lambduh)


# Combine to Do VI
vi = VI(encoder=encoder, decoder=decoder)


# Only train encoder
optimizer = t.optim.Adam(vi.encoder.parameters(), lr=0.0000001)
p_bar = tqdm(range(21))
observations = Variable(t.from_numpy(data['observations'].reshape(S, 1, n)).float())
for epoch in p_bar:
    l = 0
    optimizer.zero_grad()
    loss = -1.0*vi.elbo(observations)
    loss.backward(retain_graph=True)
    optimizer.step()
    l += 1.0
    p_bar.set_description("Epoch {0}, ELBO {1}".format(epoch, l))
    print(epoch, l)


