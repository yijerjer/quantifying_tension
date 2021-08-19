import numpy as np
import scipy.integrate as integrate
import scipy.special as special
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


def cov(X, weights=None):
    if weights is None:
        M = X - X.mean(dim=0)
        return torch.matmul(M.T, M) / (M.shape[0] - 1)
    else:
        w_sum = weights.sum()
        M = X - (X * weights[:, None]).sum() / w_sum
        return (torch.matmul((M * weights[:, None]).T, M)
                / (w_sum - (weights**2).sum() / w_sum))


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
        else:
            self.weights = weights / weights.sum()
        self.neff = 1 / (self.weights**2).sum()
        # Scott's rule
        self.bw = (self.neff ** (-1 / (self.dims + 4)))

        self.covar = torch.atleast_2d(cov(X, weights=self.weights)) * self.bw**2
        self.inv_cov = torch.inverse(self.covar)
        L = torch.linalg.cholesky(self.covar * 2 * np.pi)
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

        diffs = (self.X[:, None, :] - y)
        tdiffs = (diffs[:, :, None, :] * self.inv_cov.T).sum(3)  # matmul
        energies = torch.sum(diffs * tdiffs, dim=2).T
        log_to_sums = (2.0 * torch.log(self.weights) - self.log_det) - energies
        results = torch.logsumexp(0.5 * log_to_sums, dim=1)

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
    def __init__(self, device, n_points=1000):
        super(BayesFactorKDE, self).__init__()
        self.device = device
        self.n_points = n_points

    def forward(self, XA_1d, XB_1d, X_prior_1d, weights=None):
        if weights is not None:
            kdeA = GaussianKDE(XA_1d, self.device, weights=weights["XA"])
            kdeB = GaussianKDE(XB_1d, self.device, weights=weights["XB"])
            kde_prior = GaussianKDE(X_prior_1d, self.device,
                                    weights=weights["X_prior"])
        else:
            kdeA = GaussianKDE(XA_1d, self.device)
            kdeB = GaussianKDE(XB_1d, self.device)
            kde_prior = GaussianKDE(X_prior_1d, self.device)

        X = torch.cat((XA_1d, XB_1d, X_prior_1d), dim=0)
        lims = get_limits(X)[0]
        y = torch.linspace(lims[0], lims[1], self.n_points, device=self.device)
        interval = y[1] - y[0]
        del XA_1d, XB_1d, X_prior_1d

        log_dist_A = kdeA.log_prob(y)
        log_dist_B = kdeB.log_prob(y)
        log_dist_prior = kde_prior.log_prob(y)
        del kdeA, kdeB, kde_prior

        log_to_sum_prior = log_dist_A + log_dist_B - log_dist_prior
        logR = torch.logsumexp(
            log_to_sum_prior + torch.log(interval),
            dim=0
        )
        return logR


class SuspiciousnessKDE(nn.Module):
    def __init__(self, device, n_points=1000, logsumexp=False):
        super(SuspiciousnessKDE, self).__init__()
        self.device = device
        self.n_points = n_points
        self.logsumexp = logsumexp

    def forward(self, XA_1d, XB_1d, X_prior_1d, weights={}):
        XAB_1d = torch.cat((XA_1d, XB_1d))
        if weights:
            kdeA = GaussianKDE(XA_1d, self.device, weights=weights["XA"])
            kdeB = GaussianKDE(XB_1d, self.device, weights=weights["XB"])
            XAB_weights = torch.cat((weights["XA"], weights["XB"]))
            kdeAB = GaussianKDE(XAB_1d, self.device, weights=XAB_weights)
            kde_prior = GaussianKDE(X_prior_1d, self.device,
                                    weights=weights["X_prior"])
        else:
            kdeA = GaussianKDE(XA_1d, self.device)
            kdeB = GaussianKDE(XB_1d, self.device)
            kdeAB = GaussianKDE(XAB_1d, self.device)
            kde_prior = GaussianKDE(X_prior_1d, self.device)

        X = torch.cat((XAB_1d, X_prior_1d), dim=0)
        lims = get_limits(X)[0]
        y = torch.linspace(lims[0], lims[1], self.n_points, device=self.device)
        interval = y[1] - y[0]
        del XA_1d, XB_1d, XAB_1d, X_prior_1d

        log_dist_A = kdeA.log_prob(y)
        log_dist_B = kdeB.log_prob(y)
        log_dist_AB = kdeAB.log_prob(y)
        log_dist_prior = kde_prior.log_prob(y)
        del kdeA, kdeB, kdeAB, kde_prior

        log_to_sum_prior = log_dist_A + log_dist_B - log_dist_prior
        logR = torch.logsumexp(
            log_to_sum_prior + torch.log(interval), dim=0
        )

        kldiv_A = self.kl_divergence(log_dist_A, log_dist_prior, interval)
        kldiv_B = self.kl_divergence(log_dist_B, log_dist_prior, interval)
        kldiv_AB = self.kl_divergence(log_dist_AB, log_dist_prior, interval)
        # logI = torch.log(kldiv_A) + torch.log(kldiv_B) - torch.log(kldiv_AB)
        logI = torch.log(kldiv_A * kldiv_B / kldiv_AB)

        logS = logR - logI
        return logS

    def kl_divergence(self, log_dist_post, log_dist_prior, interval):
        # return F.kl_div(log_dist_prior, log_dist_post, log_target=True)
        to_sum = torch.exp(log_dist_post) * (log_dist_post - log_dist_prior)
        return (to_sum * interval).sum()


class BayesFactor(nn.Module):
    def __init__(self, hist_type="gaussian", hist_param=1, n_dist_bins=500,
                 n_prior_bins=50):
        super(BayesFactor, self).__init__()
        self.hist_type = hist_type
        self.hist_param = hist_param
        self.n_dist_bins = n_dist_bins
        self.n_prior_bins = n_prior_bins

    def forward(self, XA_1d, XB_1d, X_prior_1d, weights={}):
        # https://discuss.pytorch.org/t/kernel-density-estimation-as-loss-function/62261/6 
        if weights:
            weights_A, bins_A = binned_weights(XA_1d, self.n_dist_bins,
                                               self.hist_type, self.hist_param,
                                               weights=weights["XA"])
            weights_B, bins_B = binned_weights(XB_1d, self.n_dist_bins,
                                               self.hist_type, self.hist_param,
                                               weights=weights["XB"])
            weights_prior, bins_prior = binned_weights(
                X_prior_1d, self.n_prior_bins, self.hist_type, self.hist_param,
                weights=weights["X_prior"]
            )
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
    def __init__(self, hist_type="gaussian", hist_param=1, log_kldiv=False,
                 n_dist_bins=500, n_prior_bins=50, return_extras=False):
        super(SuspiciousnessKLDiv, self).__init__()
        self.hist_type = hist_type
        self.hist_param = hist_param
        self.n_dist_bins = n_dist_bins
        self.n_prior_bins = n_prior_bins
        self.return_extras = return_extras
        self.log_kldiv = log_kldiv

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
        if self.log_kldiv:
            kl_div_A = torch.log(kl_div_A)
            kl_div_B = torch.log(kl_div_B)
            kl_div_AB = torch.log(kl_div_AB)

        log_I = kl_div_A + kl_div_B - kl_div_AB

        log_S = torch.log(R) - log_I
        if self.return_extras:
            return log_S, torch.log(R), log_I, kl_div_A, kl_div_B, kl_div_AB
        else:
            return log_S, log_S

    def kl_divergence(self, post_weights, post_bins, X_prior):
        bin_width = post_bins[1] - post_bins[0]
        low_lim = post_bins[0] - bin_width / 2
        high_lim = post_bins[-1] + bin_width / 2
        prior_weights, prior_bins = binned_weights(
            X_prior, post_bins.shape[0], self.hist_type, self.hist_param,
            low_lim=low_lim, high_lim=high_lim
        )

        # post_div_prior = post_weights / prior_weights
        # kl_div = post_weights * torch.log(post_div_prior) * bin_width
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


def sigma_from_logS(d, logS):
    def chi2_pdf(x, d):
        return ((np.power(x, d/2 - 1) * np.exp(-x/2)) 
                / (np.power(2, d/2) * special.gamma(d/2)))

    lim = d[0] - 2 * logS[0]
    lim_err = d[1] + 2 * logS[1]
    p = integrate.quad(chi2_pdf, lim, 1000, args=(d[0]))
    sigma = np.sqrt(2) * special.erfinv(1 - p[0])

    d_low = d[0] - d[1]
    d_high = d[0] + d[1]
    lim_low = lim - lim_err
    lim_high = lim + lim_err

    p_ll = integrate.quad(chi2_pdf, lim_low, 1000, args=(d_low))[0]
    p_lh = integrate.quad(chi2_pdf, lim_low, 1000, args=(d_high))[0]
    p_hl = integrate.quad(chi2_pdf, lim_high, 1000, args=(d_low))[0]
    p_hh = integrate.quad(chi2_pdf, lim_high, 1000, args=(d_high))[0]
    p_vals = [p_ll, p_lh, p_hl, p_hh]
    p_err = (max(p_vals) - min(p_vals)) / 2

    sigma_low = np.sqrt(2) * special.erfinv(1 - (min(p_vals)))
    sigma_high = np.sqrt(2) * special.erfinv(1 - (max(p_vals)))
    sigma_err = abs(sigma_low - sigma_high) / 2

    return (sigma, sigma_err), (p[0], p_err)
