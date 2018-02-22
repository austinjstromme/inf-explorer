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
        self.enc_mu = nn.Linear(100, 8)
        self.enc_log_sigma = nn.Linear(100, 8)

    def _get_mu_sigma(self, x):
        h = F.relu(self.linear1(x))
        h = F.relu(self.linear2(h))
        mu = self.enc_mu(h)
        log_sigma = self.enc_log_sigma(h)
        sigma = t.exp(log_sigma)
        return mu, sigma

    def sample(self, x):
        mu, sigma = self._get_mu_sigma(x)
        std_normal = t.from_numpy(np.random.normal(0,1,size=sigma.size())).float()
        return mu + sigma * Variable(std_normal, requires_grad=False)

    def entropy(self, x):
        mu, sigma = self._get_mu_sigma(x)
        return t.mean(t.log(sigma))

class MNIST_Decoder(Decoder):
    def __init__(self, **model_params):
        super(MNIST_Decoder, self).__init__(model_params)
        # Note assigning these linear parameters to self -> adds them to parameters
        self.linear1 = nn.Linear(model_params['D_in'], model_params['H'])
        self.linear2 = nn.Linear(model_params['H'], model_params['D_out'])

    def predict_x(self, z):
        h = F.relu(self.linear1(z))
        return F.relu(self.linear2(h))

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

    encoder = MNIST_Encoder(D_in=input_dim, H=100, D_out=100)
    decoder = MNIST_Decoder(D_in=8, H=100, D_out=input_dim)
    vi = VI(encoder, decoder)

    optimizer = t.optim.Adam(vi.parameters(), lr=0.001)
    l = 0
    p_bar = tqdm(range(100))
    for epoch in p_bar:
        for i, data in enumerate(dataloader, 0):
            inputs, _ = data
            inputs = Variable(inputs.resize_(batch_size, input_dim))
            optimizer.zero_grad()
            loss = -1.0*vi.elbo(inputs)
            loss.backward()
            optimizer.step()
            l = -1.0*loss.data[0]
            p_bar.set_description("Epoch {0}, ELBO {1}".format(epoch, l))
        print(epoch, l)

        if epoch % 5 == 0:
            import matplotlib.pyplot as plt
            plt.imshow(vi.predict_x(inputs).data[0].numpy().reshape(28, 28), cmap='gray')
            plt.show(block=True)

