import torch
import torch.distributions as dists
import matplotlib.pyplot as plt
from anesthetic.plot import kde_plot_1d, kde_contour_plot_2d


def get_limits(points):
    min_max = torch.tensor([])
    for row in torch.transpose(points, 0, 1):
        min_val = torch.min(row)  
        max_val = torch.max(row)
        padding = (max_val - min_val) / 100
        min_max = torch.cat((min_max, torch.tensor([[min_val - padding, max_val + padding]])))

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

    grid_z = tension_net(grid_xy).squeeze(2)
    z = grid_z.reshape(-1)
    z_llhd = likelihood_f(z, X_combine)
    grid_z_llhd = z_llhd.reshape(grid_x.shape)

    contour_plot = axs.contour(grid_x.detach().numpy(), grid_y.detach().numpy(), grid_z_llhd.detach().numpy(), levels=30)
    kde_contour_plot_2d(axs, XA[:, 0], XA[:, 1])
    kde_contour_plot_2d(axs, XB[:, 0], XB[:, 1])
    # fig.colorbar(contour_plot)


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

    grid_z = tension_net(grid_xy).squeeze(2)

    contour_plot = axs.contour(grid_x.detach().numpy(), grid_y.detach().numpy(), grid_z.detach().numpy(), levels=30)
    kde_contour_plot_2d(axs, XA[:, 0], XA[:, 1])
    kde_contour_plot_2d(axs, XB[:, 0], XB[:, 1])
    # fig.colorbar(contour_plot, ax=axs)
