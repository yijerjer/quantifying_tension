import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dists
from torch.distributions import MultivariateNormal
from torch.distributions.distribution import Distribution
import torch.nn.functional as F
from torch_utils import get_limits


# https://discuss.pytorch.org/t/differentiable-torch-histc/25865/3
class SoftHistogram(nn.Module):
    def __init__(self, n_bins, min, max, param=None, envelope="gaussian",
                 device=None):
        super(SoftHistogram, self).__init__()
        self.n_bins = n_bins
        if param:
            self.param = param
        else:
            if envelope == "sigmoid":
                self.param = 10
            elif envelope == "gaussian":
                self.param = 1

        self.envelope = envelope
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")
        self.bin_width = (max - min) / n_bins
        self.centres = min + self.bin_width * (torch.arange(n_bins) + 1/2)
        self.centres = self.centres.to(self.device)

    def forward(self, x, weights=None):
        x = x.squeeze(1)
        x = x.unsqueeze(0) - self.centres.unsqueeze(1)
        out = self.envelope_function(x)
        if weights is not None:
            sum_weights = weights.sum()
            weights = weights / sum_weights * weights.shape[0]
            out *= weights
        out = out.sum(dim=1)
        return out, self.centres

    def envelope_function(self, x):
        if self.envelope == "sigmoid":
            return (torch.sigmoid(self.param * (x + self.bin_width / 2))
                    - torch.sigmoid(self.param * (x - self.bin_width / 2)))
        elif self.envelope == "gaussian":
            std = self.param * self.bin_width / 2
            # multiplied by self.bin_width because we want the integral of
            # it to be self.bin_width
            # Think of the top-hat in a histogram -
            # the integral of it is self.bin_width
            return ((torch.exp(-torch.square(x) / (2 * torch.square(std))))
                    / (np.sqrt(2 * np.pi) * std) * self.bin_width)


def cov(X):
    M = X - X.mean(dim=0)
    return torch.matmul(M.T, M) / (M.shape[0] - 1)


class GaussianKDE(Distribution):
    def __init__(self, X, device, weights=None):
        self.X = X
        self.device = device
        self.n = torch.tensor(X.shape[0], device=device)
        self.dims = torch.tensor(X.shape[1], device=device)
        self.mvn = MultivariateNormal(
            loc=torch.zeros(self.dims, device=device),
            covariance_matrix=torch.eye(self.dims, device=device)
        )

        if weights is None:
            self.weights = torch.ones(self.n, device=device) / self.n
        self.neff = 1 / (self.weights**2).sum()
        # Scott's rule
        self.bw = (self.n ** (-1 / (self.dims + 4)))

        self.covar = torch.atleast_2d(cov(X)) * self.bw**2
        self.inv_cov = torch.inverse(self.covar)
        L = torch.cholesky(self.covar * 2 * np.pi)
        self.log_det = 2 * torch.log(torch.diag(L)).sum()

    def prob(self, y):
        X = self.X.unsqueeze(1)
        y = y.unsqueeze(1)
        prob = (torch.exp(self.mvn.log_prob((X - y) / self.bw)).sum(dim=0)
                / (self.n * self.bw.prod()))
        return prob

    def log_prob(self, y):
        if self.dims == 1:
            y = y[:, None]

        # y_n = y.shape[0]

        diffs = (self.X[:, None, :] - y)
        tdiffs = (diffs[:, :, None, :] * self.inv_cov.T).sum(3) # matmul
        energies = torch.sum(diffs * tdiffs, dim=2).T
        log_to_sums = (2.0 * torch.log(self.weights) - self.log_det) - energies
        results = torch.logsumexp(0.5 * log_to_sums, dim=1)

        # results = torch.empty((y_n, ), dtype=torch.float, device=self.device)
        # for i in range(y_n):
        #     diff = self.X - y[i, :]
        #     tdiff = torch.matmul(diff, self.inv_cov)
        #     energy = torch.sum(diff * tdiff, dim=1)
        #     log_to_sum = 2.0 * torch.log(self.weights) - self.log_det - energy
        #     results[i] = torch.logsumexp(0.5 * log_to_sum, 0)

        return results

    # https://stats.stackexchange.com/questions/175580/how-do-i-estimate-a-smooth-cdf-from-a-set-of-observations
    def cdf(self, y):
        X = self.X.unsqueeze(1)
        y = y.unsqueeze(1)
        cdf_prob = (self.cdf_f((y - X) / self.bw).sum(dim=0) / (self.n))
        return cdf_prob

    def cdf_f(self, X):
        root_two = torch.sqrt(torch.tensor(2).float())
        return (0.5 * (1 + torch.erf(X / root_two))).prod(2)


class BayesFactorKDE(nn.Module):
    def __init__(self, device, n_points=1000, flatten_prior=False,
                 prior_division=True, return_extras=False, logsumexp=False):
        super(BayesFactorKDE, self).__init__()
        self.device = device
        self.n_points = n_points
        self.flatten_prior = flatten_prior
        self.prior_division = prior_division
        self.return_extras = return_extras
        self.logsumexp = logsumexp

    def forward(self, XA_1d, XB_1d, X_prior_1d, weights={}):
        if self.flatten_prior:
            kde_prior = GaussianKDE(X_prior_1d, self.device)
            XA_1d = kde_prior.cdf(XA_1d.squeeze())
            XB_1d = kde_prior.cdf(XB_1d.squeeze())
            X_prior_1d = kde_prior.cdf(X_prior_1d.squeeze())
            XA_1d = XA_1d.unsqueeze(1)
            XB_1d = XB_1d.unsqueeze(1)
            X_prior_1d = X_prior_1d.unsqueeze(1)
        kdeA = GaussianKDE(XA_1d, self.device)
        kdeB = GaussianKDE(XB_1d, self.device)
        kde_prior = GaussianKDE(X_prior_1d, self.device)

        X = torch.cat((XA_1d, XB_1d, X_prior_1d), dim=1)
        lims = get_limits(X)[0]
        y = torch.linspace(lims[0], lims[1], self.n_points, device=self.device)
        interval = y[1] - y[0]
        del XA_1d, XB_1d, X_prior_1d

        if self.logsumexp:
            log_dist_A = kdeA.log_prob(y)
            log_dist_B = kdeB.log_prob(y)
            log_dist_prior = kde_prior.log_prob(y)
            del kdeA, kdeB, kde_prior

            if self.prior_division:
                log_to_sum = log_dist_A + log_dist_B
            else:
                log_to_sum = log_dist_A + log_dist_B - log_dist_prior
            log_to_sum = log_to_sum
            logR = torch.logsumexp(log_to_sum + torch.log(interval), dim=0)
        else:
            dist_A = kdeA.prob(y)
            dist_B = kdeB.prob(y)
            dist_prior = kde_prior.prob(y)
            del kdeA, kdeB, kde_prior

            if self.prior_division:
                R = (dist_A * dist_B) / dist_prior
            else:
                R = dist_A * dist_B
            R = R.sum() * interval
            logR = torch.log(R)

        if self.return_extras:
            return logR, dist_A, dist_B, dist_prior, y
        else:
            return logR


class BayesFactor(nn.Module):
    def __init__(self, hist_type="gaussian", hist_param=1, n_dist_bins=500,
                 n_prior_bins=50, logsumexp=False, extra=False):
        super(BayesFactor, self).__init__()
        self.hist_type = hist_type
        self.hist_param = hist_param
        self.n_dist_bins = n_dist_bins
        self.n_prior_bins = n_prior_bins
        self.logsumexp = logsumexp
        self.extra = extra

    def forward(self, XA_1d, XB_1d, X_prior_1d, weights={}):
        # https://discuss.pytorch.org/t/kernel-density-estimation-as-loss-function/62261/6 
        if weights:
            weights_A, bins_A = binned_weights(XA_1d, self.n_dist_bins,
                                               self.hist_type, self.hist_param,
                                               weights=weights["XA"])
            weights_B, bins_B = binned_weights(XB_1d, self.n_dist_bins,
                                               self.hist_type, self.hist_param,
                                               weights=weights["XB"])
        else:
            weights_A, bins_A = binned_weights(XA_1d, self.n_dist_bins,
                                               self.hist_type, self.hist_param)
            weights_B, bins_B = binned_weights(XB_1d, self.n_dist_bins,
                                               self.hist_type, self.hist_param)

        weights_prior, bins_prior = binned_weights(
            X_prior_1d, self.n_prior_bins, self.hist_type, self.hist_param
        )

        bin_width_prior = bins_prior[1] - bins_prior[0]
        cum_weights_A = self.activation_bins(bins_A, bins_prior,
                                             bin_width_prior) * weights_A
        cum_weights_B = self.activation_bins(bins_B, bins_prior,
                                             bin_width_prior) * weights_B

        cum_weights_A = cum_weights_A.sum(1)
        cum_weights_B = cum_weights_B.sum(1)

        R = (cum_weights_A * cum_weights_B) / weights_prior
        R[R == float('inf')] = 0
        R[R != R] = 0
        if self.logsumexp:
            return R.logsumexp(0)
        else:
            if self.extra == True:
                return (torch.log(R.sum()), weights_A, bins_A, weights_B, bins_B,
                        cum_weights_A, cum_weights_B, weights_prior, bins_prior)
            else:
                return torch.log(R.sum())

    def activation_bins(self, s_bins, l_centres, l_bin_width,
                        envelope="gaussian"):
        x = s_bins.unsqueeze(0) - l_centres.unsqueeze(1)
        if self.hist_type == "sigmoid":
            return (torch.sigmoid(self.hist_param * (x + l_bin_width / 2))
                    - torch.sigmoid(self.hist_param * (x - l_bin_width / 2)))
        elif self.hist_type == "gaussian":
            std = self.hist_param * l_bin_width / 2
            return ((torch.exp(-torch.square(x) / (2 * torch.square(std))))
                    / (np.sqrt(2 * np.pi) * std) * l_bin_width)


class SuspiciousnessKLDiv(nn.Module):
    def __init__(self, hist_type="gaussian", hist_param=1,
                 n_dist_bins=500, n_prior_bins=50, return_extras=False):
        super(SuspiciousnessKLDiv, self).__init__()
        self.hist_type = hist_type
        self.hist_param = hist_param
        self.n_dist_bins = n_dist_bins
        self.n_prior_bins = n_prior_bins
        self.return_extras = return_extras

    def forward(self, XA_1d, XB_1d, X_prior_1d, weights={}):
        if weights:
            weights_A, bins_A = binned_weights(XA_1d, self.n_dist_bins,
                                               self.hist_type, self.hist_param,
                                               weights=weights["XA"])
            weights_B, bins_B = binned_weights(XB_1d, self.n_dist_bins,
                                               self.hist_type, self.hist_param,
                                               weights=weights["XB"])
        else:
            weights_A, bins_A = binned_weights(XA_1d, self.n_dist_bins,
                                               self.hist_type, self.hist_param)
            weights_B, bins_B = binned_weights(XB_1d, self.n_dist_bins,
                                               self.hist_type, self.hist_param)
        weights_prior, bins_prior = binned_weights(
            X_prior_1d, self.n_prior_bins, self.hist_type, self.hist_param)

        bounds_prior = get_limits(X_prior_1d)[0]
        bin_width_prior = ((bounds_prior[1] - bounds_prior[0])
                           / self.n_prior_bins)
        cum_weights_A = self.activation_bins(bins_A, bins_prior,
                                             bin_width_prior) * weights_A
        cum_weights_B = self.activation_bins(bins_B, bins_prior,
                                             bin_width_prior) * weights_B
        cum_weights_A = cum_weights_A.sum(1)
        cum_weights_B = cum_weights_B.sum(1)

        R = (cum_weights_A * cum_weights_B) / weights_prior
        R[R == float('inf')] = 0
        R[R != R] = 0
        R = R.sum()

        kl_div_A = self.kl_divergence(weights_A, bins_A, X_prior_1d)
        kl_div_B = self.kl_divergence(weights_B, bins_B, X_prior_1d)
        XAB_1d = torch.cat((XA_1d, XB_1d))
        if weights:
            XAB_weights = torch.cat((weights["XA"], weights["XB"]))
            weights_AB, bins_AB = binned_weights(
                XAB_1d, self.n_dist_bins, self.hist_type, self.hist_param,
                weights=XAB_weights
            )
        else:
            weights_AB, bins_AB = binned_weights(
                XAB_1d, self.n_dist_bins, self.hist_type, self.hist_param
            )
        kl_div_AB = self.kl_divergence(weights_AB, bins_AB, X_prior_1d)

        log_I = kl_div_A + kl_div_B - kl_div_AB

        log_S = torch.log(R) - log_I
        if self.return_extras:
            return log_S, torch.log(R), log_I, kl_div_A, kl_div_B, kl_div_AB
        else:
            return log_S

    def kl_divergence(self, post_weights, post_bins, X_prior):
        bin_width = post_bins[1] - post_bins[0]
        low_lim = post_bins[0] - bin_width / 2
        high_lim = post_bins[-1] + bin_width / 2
        prior_weights, prior_bins = binned_weights(
            X_prior, post_bins.shape[0], self.hist_type, self.hist_param,
            low_lim=low_lim, high_lim=high_lim
        )

        # post_div_prior = post_weights / prior_weights
        # kl_div = post_weights * torch.log(post_div_prior)
        # kl_div[kl_div != kl_div] = 0
        # kl_div = kl_div.sum()

        kl_div = F.kl_div(torch.log(prior_weights), post_weights)
        return kl_div

    def activation_bins(self, s_bins, l_centres, l_bin_width,
                        envelope="gaussian"):
        x = s_bins.unsqueeze(0) - l_centres.unsqueeze(1)
        if self.hist_type == "sigmoid":
            return (torch.sigmoid(self.hist_param * (x + l_bin_width / 2))
                    - torch.sigmoid(self.hist_param * (x - l_bin_width / 2)))
        elif self.hist_type == "gaussian":
            std = self.hist_param * l_bin_width / 2
            return ((torch.exp(-torch.square(x) / (2 * torch.square(std))))
                    / (np.sqrt(2 * np.pi) * std) * l_bin_width)


class LogSuspiciousness(nn.Module):
    def __init__(self, likelihood_cov, hist_type="gaussian", hist_param=1,
                 n_bins=500):
        super(LogSuspiciousness, self).__init__()
        self.likelihood_cov = likelihood_cov.float()
        self.hist_type = hist_type
        self.hist_param = hist_param
        self.n_bins = n_bins

    def forward(self, XA_1d, XB_1d):
        weights_A, bins_A = binned_weights(XA_1d, self.n_bins,
                                           self.hist_type, self.hist_param)
        weights_B, bins_B = binned_weights(XB_1d, self.n_bins,
                                           self.hist_type, self.hist_param)
        XAB_1d = torch.cat((XA_1d, XB_1d))
        weights_AB = (torch.cat((weights_A, weights_B))
                      / (weights_A.sum() + weights_B.sum()))
        bins_AB = torch.cat((bins_A, bins_B))

        avg_log_llhd_A = (self.log_likelihood_function(bins_A, XA_1d)
                          * weights_A).sum()
        avg_log_llhd_B = (self.log_likelihood_function(bins_B, XB_1d)
                          * weights_B).sum()
        avg_log_llhd_AB = (self.log_likelihood_function(bins_AB, XAB_1d)
                           * weights_AB).sum()

        log_S = avg_log_llhd_AB - avg_log_llhd_A - avg_log_llhd_B
        return log_S

    def log_likelihood_function(self, bins, X):
        dist = dists.Normal(bins, self.likelihood_cov)
        return dist.log_prob(bins.unsqueeze(1)).sum(0)


def binned_weights(X, n_bins, hist_type, hist_param, weights=None,
                   low_lim=None, high_lim=None):
    if low_lim and high_lim:
        bounds = torch.tensor([low_lim, high_lim])
    else:
        bounds = get_limits(X)[0]
    softhist = SoftHistogram(n_bins, bounds[0], bounds[1],
                             envelope=hist_type, param=hist_param)
    counts, bins = softhist(X, weights=weights)
    weights = counts / len(X)

    return weights, bins
