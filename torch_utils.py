import numpy as np
import torch
import torch.distributions as dists
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from anesthetic.plot import kde_plot_1d, kde_contour_plot_2d


def get_limits(points, pad_div=100):
    min_max = torch.tensor([])
    for row in torch.transpose(points, 0, 1):
        min_val = torch.min(row) 
        max_val = torch.max(row)
        if min_val == max_val:
            padding = 0.1
        else:
            padding = (max_val - min_val) / pad_div
        min_max = torch.cat(
            (min_max, torch.tensor([[min_val - padding, max_val + padding]]))
        )

    return min_max


def visualise_tension(fig, axs, tension_net, XA, XB):
    XA_tensor = torch.tensor(XA).float()
    XB_tensor = torch.tensor(XB).float()
    XA_1d = tension_net(XA_tensor)
    XB_1d = tension_net(XB_tensor)
    X_combine = torch.cat((XA_1d, XB_1d))

    def likelihood_f(z, X, cov=torch.tensor(1).float()):
        normal = dists.Normal(z.float(), cov)
        return normal.log_prob(X).sum(0)

    bounds_A = get_limits(XA_tensor)
    bounds_B = get_limits(XB_tensor)
    bounds = torch.cat((bounds_A, bounds_B), dim=1)
    low_lims = bounds.min(1).values
    up_lims = bounds.max(1).values

    x = torch.linspace(low_lims[0], up_lims[0], 100)
    y = torch.linspace(low_lims[0], up_lims[1], 100)
    grid_x, grid_y = torch.meshgrid(x, y)
    grid_xy = torch.cat((grid_x.unsqueeze(2), grid_y.unsqueeze(2)), dim=2)
    points_xy = grid_xy.view(-1, 2)

    z = tension_net(points_xy)
    z_llhd = likelihood_f(z.squeeze(1), X_combine)
    grid_z_llhd = z_llhd.view(100, 100)

    axs.contour(grid_x.detach().numpy(), grid_y.detach().numpy(),
                grid_z_llhd.detach().numpy(), levels=30)
    kde_contour_plot_2d(axs, XA[:, 0], XA[:, 1])
    kde_contour_plot_2d(axs, XB[:, 0], XB[:, 1])
    axs.set_xlim([low_lims[0].item(), up_lims[0]])
    axs.set_ylim([low_lims[1].item(), up_lims[1]])


def visualise_coordinate(fig, axs, tension_net, XA, XB):
    XA_tensor = torch.tensor(XA).float()
    XB_tensor = torch.tensor(XB).float()

    bounds_A = get_limits(XA_tensor)
    bounds_B = get_limits(XB_tensor)
    bounds = torch.cat((bounds_A, bounds_B), dim=1)
    low_lims = bounds.min(1).values
    up_lims = bounds.max(1).values

    x = torch.linspace(low_lims[0], up_lims[0], 100)
    y = torch.linspace(low_lims[0], up_lims[1], 100)
    grid_x, grid_y = torch.meshgrid(x, y)
    grid_xy = torch.cat((grid_x.unsqueeze(2), grid_y.unsqueeze(2)), dim=2)
    points_xy = grid_xy.view(-1, 2)

    z = tension_net(points_xy)
    grid_z = z.view(100, 100)

    axs.contour(grid_x.detach().numpy(), grid_y.detach().numpy(),
                grid_z.detach().numpy(), levels=30)
    kde_contour_plot_2d(axs, XA[:, 0], XA[:, 1])
    kde_contour_plot_2d(axs, XB[:, 0], XB[:, 1])
    axs.set_xlim([low_lims[0].item(), up_lims[0]])
    axs.set_ylim([low_lims[1].item(), up_lims[1]])


def plot_marginalised_dists(fig, axs, XA_1d, XB_1d, X_prior_1d,
                            flat_prior=False):
    if flat_prior:
        kde = gaussian_kde(X_prior_1d)
        X_all = np.concatenate((XA_1d, XB_1d, X_prior_1d))
        pad = (np.max(X_all) - np.min(X_all)) / 100
        x = np.linspace(np.min(X_all) - pad, np.max(X_all) + pad, 1000)
        pdf = kde(x)
        cdf = np.cumsum(pdf)
        cdf /= np.max(cdf)
        cdf_f = interp1d(x, cdf)

        XA_1d = cdf_f(XA_1d)
        XB_1d = cdf_f(XB_1d)
        X_prior_1d = cdf_f(X_prior_1d)

    kde_plot_1d(axs, XA_1d)
    kde_plot_1d(axs, XB_1d)
    kde_plot_1d(axs, X_prior_1d)
