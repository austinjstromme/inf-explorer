from __future__ import print_function
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from base_class import Decoder, Encoder, VI
from tqdm import tqdm

class MNIST_Encoder(Encoder):
    def __init__(self, **approx_params):
        super(MNIST_Encoder, self).__init__(approx_params)
        self.linear1 = nn.Linear(approx_params['D_in'], approx_params['H'])
        self.linear2 = nn.Linear(approx_params['H'], approx_params['D_out'])
        self.linear1.weight.data.normal_(0, np.sqrt(2.0/approx_params['D_in']))
        self.linear2.weight.data.normal_(0, np.sqrt(2.0/approx_params['H']))

        self.enc_mu = nn.Linear(approx_params['D_out'], approx_params['Z_dim'])
        self.enc_log_sigma = nn.Linear(approx_params['D_out'], approx_params['Z_dim'])

    def _get_mu_log_sigma(self, x):
        h1 = F.relu(self.linear1(x))
        h = F.relu(self.linear2(h1))
        self.h = h
        mu = self.enc_mu(h)
        log_sigma = self.enc_log_sigma(h)
        self.mu = mu
        self.log_sigma = log_sigma
        return mu, log_sigma

    def sample(self, x):
        mu, log_sigma = self._get_mu_log_sigma(x)
        sigma = t.exp(log_sigma)
        std_normal = t.randn(sigma.size())
        return mu + sigma * Variable(std_normal, requires_grad=False)

    def entropy(self, x):
        mu, log_sigma = self._get_mu_log_sigma(x)
        return t.mean(log_sigma)

class MNIST_Decoder(Decoder):
    def __init__(self, **model_params):
        super(MNIST_Decoder, self).__init__(model_params)
        # Note assigning these linear parameters to self -> adds them to parameters
        self.linear1 = nn.Linear(model_params['D_in'], model_params['H'])
        self.linear2 = nn.Linear(model_params['H'], model_params['D_out'])

        self.linear1.weight.data.normal_(0, np.sqrt(2.0/model_params['D_in']))
        self.linear2.weight.data.normal_(0, np.sqrt(2.0/model_params['H']))

    def predict_x(self, z):
        h = F.relu(self.linear1(z))
        return F.sigmoid(self.linear2(h))

    def loglike(self, x, z):
        """ Return loglikelihood of obs + latents
            $\log Pr(X, Z | \theta)$
        """
        x_predict = self.predict_x(z)
        loglike = -0.5*t.mean((x_predict-x)*(x_predict-x)) # Pr(X | Z)
        loglike += -0.5*t.mean(z*z) # Pr(Z) = Normal(0,1)
        return loglike

if __name__ == "__main__":
    input_dim = 28*28
    batch_size = 32

    import torchvision
    from torchvision import transforms
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    dataloader = t.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    print('Number of samples: ', len(mnist))

    encoder = MNIST_Encoder(D_in=input_dim, H=200, D_out=100, Z_dim=25)
    decoder = MNIST_Decoder(D_in=25, H=200, D_out=input_dim)
    vi = VI(encoder, decoder)

    optimizer = t.optim.Adam(vi.parameters(), lr=0.00001)
    p_bar = tqdm(range(21))
    _, plot_data = enumerate(dataloader, 0).next()
    plot_inputs, plot_labels = plot_data
    plot_inputs = Variable(plot_inputs.resize_(batch_size, input_dim))

    for epoch in p_bar:
        l = 0
        for i, data in enumerate(dataloader, 0):
            inputs, _ = data
            inputs = Variable(inputs.resize_(batch_size, input_dim))
            optimizer.zero_grad()
            loss = -1.0*vi.elbo(inputs)
            loss.backward()
            optimizer.step()
            l += -1.0*loss.data[0]
            p_bar.set_description("Epoch {0}, ELBO {1}".format(epoch, l))
        print(epoch, l)

        if epoch % 10 == 0:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(5,2)
            axes[0,0].imshow(vi.predict_x(plot_inputs).data[0].numpy().reshape(28, 28), cmap='gray')
            axes[0,1].imshow(plot_inputs.data[0].numpy().reshape(28, 28), cmap='gray')
            axes[1,0].imshow(vi.predict_x(plot_inputs).data[1].numpy().reshape(28, 28), cmap='gray')
            axes[1,1].imshow(plot_inputs.data[1].numpy().reshape(28, 28), cmap='gray')
            axes[2,0].imshow(vi.predict_x(plot_inputs).data[2].numpy().reshape(28, 28), cmap='gray')
            axes[2,1].imshow(plot_inputs.data[2].numpy().reshape(28, 28), cmap='gray')
            axes[3,0].imshow(vi.predict_x(plot_inputs).data[3].numpy().reshape(28, 28), cmap='gray')
            axes[3,1].imshow(plot_inputs.data[3].numpy().reshape(28, 28), cmap='gray')
            axes[4,0].imshow(vi.predict_x(plot_inputs).data[4].numpy().reshape(28, 28), cmap='gray')
            axes[4,1].imshow(plot_inputs.data[4].numpy().reshape(28, 28), cmap='gray')

