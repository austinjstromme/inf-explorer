import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as time
import os
import joblib

import torch as t
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from tqdm import tqdm

from base_class import VI
from encoder import (
        MeanFieldGaussian, TriDiagInvGaussian,
        elementwise_NN, dial_conv_NN, rnn_type_NN, full_connected_NN,
        )
from figure_utils import plot_encoder, plot_encoder_resid, plot_latent_state

def get_encoder(encoder_nn_type, encoder_q_type, S=100):
    """ Helper for Encoder
    Args:
        encoder_nn_type (string):
            Tanh - elementwise tanh
            SELU - elementwise selu
            DilatedCNN - dialted CNN
            SimpleRNN - simple RNN
            LSTM - LSTM
            FullSELU - fully connected SELU
        encoder_q_type (string):
            MF - mean field Gaussian
            TriDiag - Tridiagonal Precision Gaussian
        S (int): length of time series (optional)
    """
    if encoder_nn_type == "Tanh":
        nn_helper = lambda: elementwise_NN(input_dim=1, latent_dim=1,
                h_dims=[10, 10], layerNN=t.nn.Tanh)

    elif encoder_nn_type == "SELU":
        nn_helper = lambda: elementwise_NN(input_dim=1, latent_dim=1,
                h_dims=[10, 10], layerNN=t.nn.SELU)

    elif encoder_nn_type == "DilatedCNN":
        nn_helper = lambda: dial_conv_NN(input_dim=1, latent_dim=1,
                h_dims=[10, 10, 10], kernel_sizes=[3,3,3], dilations=[1,2,4],
                layerNN=t.nn.SELU)

    elif encoder_nn_type == "SimpleRNN":
        nn_helper = lambda: rnn_type_NN(input_dim=1, latent_dim=1,
                h_dims=[3, 3], num_layers=[1, 1], bidirectional=True,
                layerNN=t.nn.RNN)

    elif encoder_nn_type == "LSTM":
        nn_helper = lambda: rnn_type_NN(input_dim=1, latent_dim=1,
                h_dims=[3, 3], num_layers=[1, 1], bidirectional=True,
                layerNN=t.nn.LSTM)

    elif encoder_nn_type == "FullSELU":
        nn_helper = lambda: full_connected_NN(S, h_dims = [S, S],
                layerNN=t.nn.SELU)

    else:
        raise ValueError("Unrecognized encoder_nn_type {0}".foramt(
            encoder_nn_type))

    if encoder_q_type == "MF":
        nn_mu = nn_helper()
        nn_log_lambduh = nn_helper()
        encoder = MeanFieldGaussian(nn_mu, nn_log_lambduh)

    elif encoder_q_type == "TriDiag":
        nn_mu = nn_helper()
        nn_log_alpha = nn_helper()
        nn_beta = nn_helper()
        encoder = TriDiagInvGaussian(nn_mu, nn_log_alpha, nn_beta)

    else:
        raise ValueError("Unrecognized encoder_q_type {0}".format(
            encoder_q_type))

    return encoder


def train_encoder_vi(encoder, decoder, alg_name, lr,
        trainX, trainZ, testX, testZ,
        batch_size=16, num_epochs=51, checkpoint=10,
        path_to_out="./", path_to_figs="./",
        max_time=300,
        is_SLDS=False, trainW=None, testW=None):
    """ Train VI

    Returns:
        df (pd.DataFrame): ELBO on train + test

    """
    S, n = trainX.shape[1], trainX.shape[2]

    # Setup VI
    vi = VI(encoder=encoder, decoder=decoder)

    # Only train encoder
    optimizer = t.optim.Adam(vi.encoder.parameters(), lr=lr)

    # Setup Data for training + evaluation
    dataset = TensorDataset(t.from_numpy(trainX).float(),
            t.from_numpy(trainZ).float())
    dataloader = t.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=True)
    var_trainX = Variable(t.from_numpy(trainX).float())
    var_trainZ = Variable(t.from_numpy(trainZ).float())
    var_testX = Variable(t.from_numpy(testX).float())
    var_testZ = Variable(t.from_numpy(testZ).float())

    df = pd.DataFrame()
    elapsed_time = 0.0
    train_elbo = vi.elbo(var_trainX).data[0]
    test_elbo = vi.elbo(var_testX).data[0]
    df = df.append({
        'epoch': 0,
        'train_elbo': train_elbo,
        'test_elbo': test_elbo,
        'time': elapsed_time,
        }, ignore_index=True)

    # Training Loop
    print("Training alg_name: {0}".format(alg_name))
    p_bar = tqdm(range(1, num_epochs+1))
    for epoch in p_bar:
        elbo_hat, counts = 0.0, 0.0
        start_time = time.time()
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
        elapsed_time += time.time() - start_time
        train_elbo = vi.elbo(var_trainX).data[0]
        test_elbo = vi.elbo(var_testX).data[0]
        progress = {
            'epoch': epoch,
            'train_elbo': train_elbo,
            'test_elbo': test_elbo,
            'time': elapsed_time,
            }
        print(progress)
        df = df.append(progress, ignore_index=True)

        if epoch % checkpoint == 1 or epoch <= 10:
            # Save df to output
            df.to_csv(os.path.join(path_to_out,"{0}_df.csv".format(alg_name)),
                    index=False)

            # Save VI to joblib
            joblib.dump(vi, os.path.join(path_to_out, "{0}_vi.p".format(alg_name)))

            plt.close('all')
            if not is_SLDS:
                # Make Train Figs
                fig, ax = plt.subplots(1,1)
                fig.set_size_inches(8, 8)
                plot_encoder(vi.encoder, trainX[0], trainZ[0],
                        ax=ax)
                ax.set_title("{1}, Epoch={0}".format(epoch, alg_name))
                fig.savefig(os.path.join(path_to_figs,
                    "{0}_epoch{1}_train.png".format(alg_name, epoch)))

                # Make Test Fit Figs
                fig, ax = plt.subplots(1,1)
                fig.set_size_inches(8, 8)
                plot_encoder(vi.encoder, testX[0], testZ[0],
                        ax=ax)
                ax.set_title("{1}, Epoch={0}".format(epoch, alg_name))
                fig.savefig(os.path.join(path_to_figs,
                    "{0}_epoch{1}_test.png".format(alg_name, epoch)))
                plt.close('all')
            else:
                # Make Train Figs
                fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios':[5,1]})
                fig.set_size_inches(8, 8)
                plot_encoder(vi.encoder, trainX[0], trainZ[0],
                        ax=ax[0])
                ax[0].set_title("{1}, Epoch={0}".format(epoch, alg_name))
                plot_latent_state(trainW[0], ax=ax[1])
                ax[1].set_ylabel("Discrete Latent State")
                fig.savefig(os.path.join(path_to_figs,
                    "{0}_epoch{1}_train.png".format(alg_name, epoch)))

                # Make Test Fit Figs
                fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios':[5,1]})
                fig.set_size_inches(8, 8)
                plot_encoder(vi.encoder, testX[0], testZ[0],
                        ax=ax[0])
                ax[0].set_title("{1}, Epoch={0}".format(epoch, alg_name))
                plot_latent_state(testW[0], ax=ax[1])
                ax[1].set_ylabel("Discrete Latent State")
                fig.savefig(os.path.join(path_to_figs,
                    "{0}_epoch{1}_test.png".format(alg_name, epoch)))
                plt.close('all')

        if elapsed_time > max_time:
            break

    return df

