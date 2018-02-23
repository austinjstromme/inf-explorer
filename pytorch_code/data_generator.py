"""
LDS Data Generator
"""
import numpy as np
import logging

logger = logging.getLogger(name=__name__)

def generate_lds_data(T, A, Q, C, R, initial_message = None):
    """ Helper function for generating LDS data

    Args:
        T (int): length of series
        A (n by n ndarray): latent state transition
        Q (n by n ndarray): latent state noise
        C (m by n ndarray): emission matrix
        R (m by m ndarray): emission noise
        initial_message (dict, optional): prior for x_{-1}
            log_constant (double)
            mean_precision (n ndarray)
            precision (n by n ndarray)

    Returns:
        data (dict): dictionary containing:
            observations (ndarray): T by m
            latent_vars (ndarray): T by n
            parameters (dict): A, Q, C, R
            initial_message (dict):
    """
    m, n = np.shape(C)
    parameters = dict(A=A, C=C, Q=Q, R=R)
    if initial_message is None:
        initial_message = dict(
                log_constant = 0.0,
                mean_precision = np.zeros(n),
                precision = np.eye(n)/10.0,
                )
    latent_vars = np.zeros((T, n)) # Latent States
    obs_vars = np.zeros((T, m)) # Observations

    latent_prev = np.random.multivariate_normal(
            mean = np.linalg.solve(
                initial_message['precision'],
                initial_message['mean_precision'],
                ),
            cov = np.linalg.inv(
                initial_message['precision'],
                )
            )
    for t in range(0,T):
        latent_vars[t] = A.dot(latent_prev) + \
            np.random.multivariate_normal(np.zeros(n), cov = Q)
        obs_vars[t] = C.dot(latent_vars[t]) + \
            np.random.multivariate_normal(np.zeros(m), cov = R)
        latent_prev = latent_vars[t]

    data = dict(
            observations = obs_vars,
            latent_vars = latent_vars,
            parameters = parameters,
            initial_message = initial_message,
            )
    return data




