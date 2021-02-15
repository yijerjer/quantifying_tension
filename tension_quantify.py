import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dists
from torch_utils import get_limits


# https://discuss.pytorch.org/t/differentiable-torch-histc/25865/3
class SoftHistogram(nn.Module):
    def __init__(self, n_bins, min, max, param=None,
                 envelope="gaussian", device=None):
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

    def forward(self, x):
        x = x.squeeze(1)
        x = x.unsqueeze(0) - self.centres.unsqueeze(1)
        out = self.envelope_function(x)
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


class BayesFactor(nn.Module):
    def __init__(self, hist_type="gaussian", hist_param=1,
                 n_dist_bins=500, n_prior_bins=50):
        super(BayesFactor, self).__init__()
        self.hist_type = hist_type
        self.hist_param = hist_param
        self.n_dist_bins = n_dist_bins
        self.n_prior_bins = n_prior_bins

    def forward(self, XA_1d, XB_1d, X_prior_1d):
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
    def __init__(self, hist_type="gaussian", hist_param=1,
                 n_dist_bins=500, n_prior_bins=50):
        super(SuspiciousnessKLDiv, self).__init__()
        self.hist_type = hist_type
        self.hist_param = hist_param
        self.n_dist_bins = n_dist_bins
        self.n_prior_bins = n_prior_bins
        self.kldiv = nn.KLDivLoss()

    def forward(self, XA_1d, XB_1d, X_prior_1d):
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
        weights_AB, bins_AB = binned_weights(XAB_1d, self.n_dist_bins,
                                             self.hist_type, self.hist_param)
        kl_div_AB = self.kl_divergence(weights_AB, bins_AB, X_prior_1d)

        log_I = kl_div_A + kl_div_B - kl_div_AB

        log_S = torch.log(R) - log_I
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

        kl_div = self.kldiv(post_bins, post_weights)
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


def binned_weights(X, n_bins, hist_type, hist_param,
                   low_lim=None, high_lim=None):
    if low_lim and high_lim:
        bounds = torch.tensor([low_lim, high_lim])
    else:
        bounds = get_limits(X)[0]
    softhist = SoftHistogram(n_bins, bounds[0], bounds[1],
                             envelope=hist_type, param=hist_param)
    counts, bins = softhist(X)
    weights = counts / len(X)

    return weights, bins
