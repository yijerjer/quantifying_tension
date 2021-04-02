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


def planck_des_data(size=10000, params=None, only_lcdm=True, div_max=False,
                    mean="DES_planck"):
    planck_root = os.path.join("runs_default", "chains", "planck")
    planck_samples = NestedSamples(root=planck_root, label="Planck")
    DES_root = os.path.join("runs_default", "chains", "DES")
    DES_samples = NestedSamples(root=DES_root, label="DES")
    prior_samples = DES_samples.set_beta(beta=0)
    planck_root = os.path.join("runs_default", "chains", "DES_planck")
    Dp_samples = NestedSamples(root=planck_root, label="DES_planck")

    planck_samples = (planck_samples[planck_samples['weight'] > 10e-3])
    DES_samples = DES_samples[DES_samples['weight'] > 10e-3]
    prior_samples = prior_samples[prior_samples['weight'] > 10e-3]

    if params is None:
        if only_lcdm:
            params = ["omegabh2", "omegach2", "theta", "tau", "logA", "ns"]
        else:
            params = ["omegabh2", "omegach2", "theta", "tau", "logA", "ns",
                      "age", "omegam", "omegamh2", "H0", "sigma8", "S8"]

    X0 = np.array(planck_samples[params])
    X1 = np.array(DES_samples[params])
    X0_weights = np.array(planck_samples["weight"])
    X1_weights = np.array(DES_samples["weight"])
    X_prior = np.array(prior_samples[params])
    X_prior_weights = np.array(prior_samples["weight"])

    if mean == "DES_planck":
        X_Dp = np.array(Dp_samples[params])
        X_Dp_weights = np.array(Dp_samples["weight"])
        param_means = X_Dp * X_Dp_weights[:, None]
        param_means = param_means.sum(axis=0) / X_Dp_weights.sum()
    elif mean == "planck":
        param_means = X0 * X0_weights[:, None]
        param_means = param_means.sum(axis=0) / X0_weights.sum()

    norm_factors = np.ones(X0.shape[1])
    if div_max:
        X_combine = np.concatenate((X0, X1), axis=0)
        norm_factors = X_combine.max(axis=0)
        X0 /= norm_factors
        X1 /= norm_factors
        X_prior /= norm_factors
        param_means /= norm_factors

    return (X0, X0_weights, X1, X1_weights, X_prior, X_prior_weights,
            params, param_means, norm_factors)


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
