import os
import numpy as np
from anesthetic import NestedSamples


def simple_data(size=10000, dims=2, distance=5, fixed=False):
    single_dim_dist = distance / np.sqrt(dims)
    if fixed:
        mu0 = np.full(dims, 5.0)
        mu1 = np.full(dims, 5.0)
        mu1 += single_dim_dist
    else:
        mu0 = np.random.uniform(size=dims, low=5, high=6)
        mu1 = np.full(dims, 6.0)
        mu1[:2] += single_dim_dist
    Sigma = np.identity(dims)

    X0 = np.random.multivariate_normal(mu0, Sigma, size=(size))
    X1 = np.random.multivariate_normal(mu1, Sigma, size=(size))

    X_all = np.concatenate((X0, X1))
    prior_limits = get_limits(X_all)
    X_prior = uniform_prior_samples(prior_limits)

    return X0, X1, X_prior


def curved_data(size=10000, radius=8, dims=2):
    mu0 = np.concatenate((np.full(2, 5), np.random.uniform(size=dims-2, low=5, high=6)))
    Sigma0 = np.identity(dims)
    X0 = np.random.multivariate_normal(mu0, Sigma0, size=(size))

    mu1 = np.concatenate((np.full(2, 5), np.full(dims - 2, 6)))
    Sigma1 = np.identity(dims)
    thetas = np.random.uniform(low=0, high=np.pi / 2, size=size)
    thetas_xy = np.transpose(np.vstack((np.cos(thetas), np.sin(thetas))))
    thetas_xy *= radius
    X1 = np.random.multivariate_normal(mu1, Sigma1, size=(size))
    X1[:, :2] += thetas_xy

    X_all = np.concatenate((X0, X1))
    prior_limits = get_limits(X_all)
    X_prior = uniform_prior_samples(prior_limits)

    return X0, X1, X_prior


def planck_des_data(size=10000):
    planck_root = os.path.join("runs_default", "chains", "planck")
    planck_samples = NestedSamples(root=planck_root, label="Planck")
    DES_root = os.path.join("runs_default", "chains", "DES")
    DES_samples = NestedSamples(root=DES_root, label="DES")

    planck_samples = (planck_samples[planck_samples['weight'] > 10e-4]
                      .sample(n=size))
    DES_samples = DES_samples[DES_samples['weight'] > 10e-4].sample(n=size)

    params = ["omegabh2", "omegach2", "theta", "tau", "logA", "ns", "age",
              "omegal", "omegam", "omegamh2", "H0", "sigma8", "S8"]
    # params = ["omegabh2", "omegach2", "theta", "tau", "logA", "ns", "age"]
    X0 = np.array(planck_samples[params])
    X1 = np.array(DES_samples[params])
    X0_weights = np.array(planck_samples["weight"])
    X1_weights = np.array(DES_samples["weight"])

    X_all = np.concatenate((X0, X1))
    prior_limits = get_limits(X_all)
    X_prior = uniform_prior_samples(prior_limits)

    return X0, X0_weights, X1, X1_weights, X_prior, params


def get_limits(points):
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    padding = (maxs - mins) / 10
    padding[padding == 0] = 0.1

    mins = np.expand_dims(mins - padding, axis=1)
    maxs = np.expand_dims(maxs + padding, axis=1)
    min_max = np.concatenate((mins, maxs), axis=1)

    return min_max


def uniform_prior_samples(limits, size=10000):
    return np.random.uniform(limits[:, 0], limits[:, 1],
                             size=(size, limits.shape[0]))
