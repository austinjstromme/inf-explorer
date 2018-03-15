#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


import torch as t
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from tqdm import tqdm

from base_class import VI
from encoder import (
        MeanFieldGaussian, elementwise_MFGaussian, dial_conv_MFGaussian,
        TriDiagInvGaussian, elementwise_NN, dial_conv_NN, rnn_type_NN,
        full_connected_NN,
        )
from decoder import SLDSDecoder
from data_generator import generate_lds_data
from figure_utils import plot_encoder, plot_encoder_resid
from experiment_utils import get_encoder, train_encoder_vi


np.random.seed(1234)
t.manual_seed(1234)

n = 1 # Dimension of x_s
m = 1 # Dimension of z_s
batch_size = 16 # Batchsize
N = 256 # Number of replications
S = 100 # Length of Timeseries


# LDS Model
print("Setting Up SLDS Model Decoder")
Pi = np.array([[0.95, 0.05, 0.0], [0.05, 0.9, 0.05], [0.0, 0.05, 0.95]])
A = np.array([np.eye(m)*0.99, np.eye(m)*0.99, np.eye(m)*0.99])
Delta_Q = np.array([np.ones(m), np.zeros(m), -np.ones(m)])*0.1
Q = np.array([np.eye(m), np.eye(m), np.eye(m)])*0.1
C = np.array([np.eye(n), np.eye(n), np.eye(n)])
R = np.array([np.eye(n), np.eye(n), np.eye(n)])*1.0
num_states = np.shape(A)[0]

decoder = SLDSDecoder(num_states=num_states, m=m, n=n, S=S, C=C, R=R, A=A, Q=Q,
        Delta_Q=Delta_Q, Pi=Pi)
print("Generating Data")
trainX, trainZ, trainW = decoder.generate_data(N=N)
testX, testZ, testW = decoder.generate_data(N=N)

# VI approx
print("Setting Up VI Encoders")
list_encoders = [
        'SELU', 'DilatedCNN', 'SimpleRNN', 'LSTM', 'FullSELU',
        ]
encoders = {
        "{0}_{1}".format(encoder_nn_type, encoder_q_type): get_encoder(
            encoder_nn_type, encoder_q_type)
        for encoder_nn_type in list_encoders
        for encoder_q_type in ['MF', 'TriDiag']
        }

# Setup Experiment Output
experiment_name = "SLDSExperiment"
path_to_out= os.path.join(experiment_name,"out")
if not os.path.isdir(path_to_out):
    os.makedirs(path_to_out)

#for alg_name in encoders.keys():
agg_df = pd.DataFrame()

for alg_name in encoders.keys():
    df = pd.DataFrame()
    path_to_figs = os.path.join(experiment_name,"figs", alg_name)
    if not os.path.isdir(path_to_figs):
        os.makedirs(path_to_figs)
    lr = 0.001
    if alg_name in ["FullSELU_TriDiag", "SimpleRNN_TriDiag", "SELU_TriDiag"]:
        lr = 0.0001
    try:
        df = train_encoder_vi(encoders[alg_name], decoder, alg_name,
            lr=lr, trainX=trainX, trainZ=trainZ, testX=testX, testZ=testZ,
            num_epochs=101,
            is_SLDS=True, trainW=trainW, testW=testW,
            path_to_out=path_to_out, path_to_figs=path_to_figs)
    except:
        print("Error with learning rate {0} for {1}".format(lr, alg_name))

    df['alg_name'] = alg_name
    agg_df = agg_df.append(df, ignore_index=True)
    agg_df.to_csv(os.path.join(path_to_out, "agg_df.csv"))

fig, axes = plt.subplots(2,2, sharey=True)
for alg_name in agg_df.alg_name.unique():
    mask = agg_df.alg_name == alg_name
    axes[0,0].plot(
        agg_df['epoch'][mask], agg_df['train_elbo'][mask],
        "-o", markersize=3, label=alg_name)
    axes[0,0].set_ylabel("Train ELBO")
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].legend()

    axes[0,1].plot(
        agg_df['epoch'][mask], agg_df['test_elbo'][mask],
        "-o", markersize=3, label=alg_name)
    axes[0,1].set_ylabel("Test ELBO")
    axes[0,1].set_xlabel("Epoch")
    axes[0,1].legend()

    axes[1,0].plot(
        agg_df['time'][mask], agg_df['train_elbo'][mask],
        "-o", markersize=3, label=alg_name)
    axes[1,0].set_ylabel("Train ELBO")
    axes[1,0].set_xlabel("Time")
    axes[1,0].legend()
    axes[1,1].plot(
        agg_df['time'][mask], agg_df['test_elbo'][mask],
        "-o", markersize=3, label=alg_name)
    axes[1,1].set_ylabel("Test ELBO")
    axes[1,1].set_xlabel("Time")
    axes[1,1].legend()
fig.set_size_inches(8, 8)
fig.savefig(os.path.join(path_to_out, "metric_fig.png"))
plt.close('all')


