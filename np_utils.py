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


def curved_data(size=10000, radius=8):
    mu0 = np.array([1, 1])
    Sigma0 = np.array([[1, 0], [0, 1]])
    X0 = np.random.multivariate_normal(mu0, Sigma0, size=(size))

    mu1 = np.array([0, 0])
    Sigma1 = np.array([[1, 0], [0, 1]])
    thetas = np.random.uniform(low=(np.pi / 24), high=(11 * np.pi / 24), size=size)
    thetas_xy = np.transpose(np.vstack((np.cos(thetas), np.sin(thetas))))
    thetas_xy *= radius
    X1 = np.random.multivariate_normal(mu1, Sigma1, size=(size))
    X1 += thetas_xy

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