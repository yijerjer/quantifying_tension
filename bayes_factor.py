import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dists
from utils import get_limits


# https://discuss.pytorch.org/t/differentiable-torch-histc/25865/3
class SoftHistogram(nn.Module):
    def __init__(self, n_bins, min, max, steepness=100):
        super(SoftHistogram, self).__init__()
        self.n_bins = n_bins
        self.steepness = steepness
        self.bin_width = (max - min) / n_bins
        self.centres = min + self.bin_width * (torch.arange(n_bins) + 1/2)
    
    def forward(self, x):
        x = x.squeeze(1)
        x = x.unsqueeze(0) - self.centres.unsqueeze(1)
        out = (torch.sigmoid(self.steepness * (x + self.bin_width / 2))
               - torch.sigmoid(self.steepness * (x - self.bin_width / 2)))
        out = out.sum(dim=1)
        return out, self.centres


class BayesFactor(nn.Module):
    def __init__(self, top_hat_steepness=1000):
        super(BayesFactor, self).__init__()
        self.steepness = top_hat_steepness

    def forward(self, XA_1d, XB_1d, X_prior_1d):
        n_dist_bins = 500
        n_prior_bins = 50

        weights_A, bins_A = self.binned_weights(XA_1d, n_dist_bins)
        weights_B, bins_B = self.binned_weights(XB_1d, n_dist_bins)
        weights_prior, bins_prior = self.binned_weights(X_prior_1d, n_prior_bins)

        bounds_prior = get_limits(X_prior_1d)[0]
        bin_width_prior = (bounds_prior[1] - bounds_prior[0]) / n_prior_bins
        cum_weights_A = self.activation_bins(bins_A, bins_prior, bin_width_prior) * weights_A
        cum_weights_B = self.activation_bins(bins_B, bins_prior, bin_width_prior) * weights_B
        cum_weights_A = cum_weights_A.sum(1)
        cum_weights_B = cum_weights_B.sum(1)

        R = (cum_weights_A * cum_weights_B) / weights_prior
        return R.sum()

    def binned_weights(self, X, n_bins):
        bounds = get_limits(X)[0]
        softhist = SoftHistogram(n_bins, bounds[0], bounds[1], self.steepness)
        counts, bins = softhist(X)
        weights = counts / len(X)

        return weights, bins

    def activation_bins(self, s_bins, l_centres, l_bin_width):
        x = s_bins.unsqueeze(0) - l_centres.unsqueeze(1)
        activation = (torch.sigmoid(self.steepness * (x + l_bin_width / 2))
                      - torch.sigmoid(self.steepness * (x - l_bin_width / 2)))
        return activation


class LogSuspiciousness(nn.Module):
    def __init__(self, likelihood_mean, likelihood_cov, top_hat_steepness=1000, n_bins=500):
        super(LogSuspiciousness, self).__init__()
        self.likelihood_mean = likelihood_mean.float()
        self.likelihood_cov = likelihood_cov.float()
        self.steepness = top_hat_steepness
        self.n_bins = n_bins

    def forward(self, XA_1d, XB_1d):
        weights_A, bins_A = self.binned_weights(XA_1d, self.n_bins)
        weights_B, bins_B = self.binned_weights(XB_1d, self.n_bins)
        XAB_1d = torch.cat(XA_1d, XB_1d)
        weights_AB, bins_AB = self.binned_weights(XAB_1d, self.n_bins)

        avg_llhd_A = (self.log_likelihood_function(bins_A) * weights_A).sum()
        avg_llhd_B = (self.log_likelihood_function(bins_B) * weights_B).sum()
        avg_llhd_AB = (self.log_likelihood_function(bins_AB) * weights_AB).sum()

        log_S = avg_llhd_AB - avg_llhd_A - avg_llhd_B
        return log_S

    def log_likelihood_function(self, X):
        dist = dists.Normal(self.likelihood_mean, self.likelihood_cov)
        return dist.log_prob(X)
