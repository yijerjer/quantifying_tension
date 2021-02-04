import numpy as np
import torch


def simple_data(size=10000):
    mu0 = np.array([1, 1])
    mu1 = np.array([5, 5])
    Sigma0 = np.array([[1, 0], [0, 1]])
    Sigma1 = np.array([[1, 0], [0, 1]])

    X0 = np.random.multivariate_normal(mu0, Sigma0, size=(size))
    X1 = np.random.multivariate_normal(mu1, Sigma1, size=(size))

    X_all = torch.tensor(np.concatenate((X0, X1)))
    prior_limits = get_limits(X_all)
    X_prior = uniform_prior_samples(prior_limits)

    return X0, X1, X_prior


def curved_data(size=10000):
    mu0 = np.array([1, 1])
    Sigma0 = np.array([[1, 0], [0, 1]])

    thetas = np.linspace(np.pi / 24, 11 * np.pi / 24, 5)
    mu1s = np.array([[5 * np.cos(t), 5 * np.sin(t)] for t in thetas])
    # mu1 = np.array([5, 5])
    Sigma1 = np.array([[0.2, 0], [0.2, 1]])

    X0 = np.random.multivariate_normal(mu0, Sigma0, size=(size))
    X1 = np.empty((0, 2))
    for mu in mu1s:
        X1 = np.concatenate((X1, np.random.multivariate_normal(mu, Sigma1, size=(int(size / 5)))))
    # X1 = np.random.multivariate_normal(mu1, Sigma1, size=(size))

    X_all = torch.tensor(np.concatenate((X0, X1)))
    prior_limits = get_limits(X_all)
    X_prior = uniform_prior_samples(prior_limits)

    return X0, X1, X_prior


def get_limits(points):
    min_max = []
    for row in np.transpose(points):
        min_val = min(row)
        max_val = max(row)
        padding = (max_val - min_val) / 10
        min_max.append([min_val - padding, max_val + padding])

    return min_max


def uniform_prior_samples(limits, size=10000):
    prior_samples = []
    for limit in limits:
        low_lim = limit[0].detach().numpy()
        high_lim = limit[1].detach().numpy()
        prior_samples.append(np.random.uniform(low_lim, high_lim, size=size))

    prior_samples = np.array(prior_samples)
    return np.transpose(prior_samples)