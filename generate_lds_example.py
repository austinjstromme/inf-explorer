# Example
import matplotlib.pyplot as plt
import numpy as np

from data_generator import generate_lds_data

T = 100
A = np.array([[0.9]])
C = np.array([[1.0]])
Q = np.array([[0.01]])
R = np.array([[1.0]])


data = generate_lds_data(T=T, A=A, Q=Q, C=C, R=R)

plt.close('all')
plt.plot(data['latent_vars'], label='latent_vars')
plt.plot(data['observations'], label='observations')
plt.legend()
plt.show()


