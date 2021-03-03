import numpy as np
import torch
import torch.distributions as dists
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from anesthetic.plot import kde_plot_1d, kde_contour_plot_2d
from tension_net import TensionNet


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


def rotation_test(net, criterion, device, XA, XB, X_prior, n_points=100):
    if isinstance(net, TensionNet):
        thetas = np.linspace(0, np.pi * 2, n_points)
        losses = []
        Xs = []
        XA_tnsr = torch.tensor(XA).to(device).float()
        XB_tnsr = torch.tensor(XB).to(device).float()
        X_prior_tnsr = torch.tensor(X_prior).to(device).float()

        for theta in thetas:
            weight = torch.tensor([[np.cos(theta), np.sin(theta)]]).to(device)
            tension_net = TensionNet(2).to(device)
            tension_net.state_dict()["linear.weight"].copy_(weight)
            tension_net.state_dict()["linear.bias"].copy_(torch.tensor([0])
                                                          .to(device))
            XA_1d = tension_net(XA_tnsr)
            XB_1d = tension_net(XB_tnsr)
            X_prior_1d = tension_net(X_prior_tnsr)

            criterion = criterion.to(device)
            loss = criterion(XA_1d, XB_1d, X_prior_1d)
            losses.append(loss.item())
            Xs.append((XA_1d, XB_1d, X_prior_1d))

        return thetas, losses, Xs
    else:
        raise ValueError("net is not of class TensionNet")


class TrainUtil:
    def __init__(self, net, optimizer, criterion, device, animation=False):
        self.net = net.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.losses = []
        self.animation = animation
        self.input_dim = self.net.input_size
        if self.animation:
            self.coordinates = []
            self.kde_dists = []

    def train(self, XA, XB, X_prior, weights={}, n_iter=500):
        self.XA = XA
        self.XB = XB
        self.X_prior = X_prior

        self.XA_tnsr = torch.tensor(XA).to(self.device).float()
        self.XB_tnsr = torch.tensor(XB).to(self.device).float()
        self.X_prior_tnsr = torch.tensor(X_prior).to(self.device).float()

        if weights:
            weights["XA"] = torch.tensor(weights["XA"]).to(self.device).float()
            weights["XB"] = torch.tensor(weights["XB"]).to(self.device).float()
            self.weights = weights
        else:
            self.weights = None

        bounds = self.get_bounds()

        for i in range(n_iter):
            self.optimizer.zero_grad()
            XA_1d = self.net(self.XA_tnsr)
            XB_1d = self.net(self.XB_tnsr)
            X_prior_1d = self.net(self.X_prior_tnsr)

            loss = self.criterion(XA_1d, XB_1d, X_prior_1d, weights=weights)
            self.losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            if self.animation:
                x, y, z = self.grid_xyz(bounds[0], bounds[1])
                self.coordinates.append((x.cpu().detach().numpy(),
                                         y.cpu().detach().numpy(),
                                         z.cpu().detach().numpy()))

                XA_1d = XA_1d.squeeze().cpu().detach().numpy()
                XB_1d = XB_1d.squeeze().cpu().detach().numpy()
                X_prior_1d = X_prior_1d.squeeze().cpu().detach().numpy()
                XA_1d, XB_1d, X_prior_1d = self.flatten_prior(XA_1d, XB_1d,
                                                              X_prior_1d)
                kde_A = self.kde_dist_1d(XA_1d)
                kde_B = self.kde_dist_1d(XB_1d)
                kde_prior = self.kde_dist_1d(X_prior_1d)
                self.kde_dists.append((kde_A, kde_B, kde_prior))

        return self.losses

    def kde_dist_1d(self, X):
        low_lim = np.min(X)
        up_lim = np.max(X)
        padding = (up_lim - low_lim) / 100

        kde = gaussian_kde(X)
        x = np.linspace(low_lim - padding, up_lim + padding, 1000)
        y = kde(x)

        x = np.expand_dims(x, 1)
        y = np.expand_dims(y, 1)
        return np.concatenate((x, y), axis=1)

    def save_animation_data(self, filename):
        np.savez(filename,
                 losses=np.array(self.losses),
                 coords=np.array(self.coordinates),
                 kdes=np.array(self.kde_dists))

    def plot_loss(self, axs):
        axs.plot(np.arange(len(self.losses)), self.losses)
        axs.set_title(f"Loss (final loss = {round(self.losses[-1], 4)})")

    def visualise_tension(self, axs, idxs=(0, 1), default_val=0):
        XA_1d = self.net(self.XA_tnsr)
        XB_1d = self.net(self.XB_tnsr)
        X_combine = torch.cat((XA_1d, XB_1d))

        low_lims, up_lims = self.get_bounds()

        def likelihood_f(z, X_1d, cov=torch.tensor(1).float().to(self.device)):
            normal = dists.Normal(z.float(), cov)
            return normal.log_prob(X_1d).sum(0)

        grid_x, grid_y, grid_z = self.grid_xyz(low_lims, up_lims, idxs=idxs,
                                               default_val=default_val)

        z = grid_z.view(-1)
        z_llhd = likelihood_f(z, X_combine)
        grid_z_llhd = z_llhd.view(100, 100)

        grid_x = grid_x.cpu().detach().numpy()
        grid_y = grid_y.cpu().detach().numpy()
        grid_z_llhd = grid_z_llhd.cpu().detach().numpy()
        axs.contour(grid_x, grid_y, grid_z_llhd, levels=30)

        kde_contour_plot_2d(axs, self.XA[:, idxs[0]], self.XA[:, idxs[1]],
                            weights=(self.weights["XA"].cpu().detach().numpy()
                                     if self.weights else None))
        kde_contour_plot_2d(axs, self.XB[:, idxs[0]], self.XB[:, idxs[1]],
                            weights=(self.weights["XB"].cpu().detach().numpy()
                                     if self.weights else None))

        axs.set_xlim([np.min(grid_x), np.max(grid_x)])
        axs.set_ylim([np.min(grid_y), np.max(grid_y)])
        axs.set_title("Likelihood contour")

    def visualise_coordinate(self, axs, idxs=(0, 1), default_val=0):
        low_lims, up_lims = self.get_bounds()

        grid_x, grid_y, grid_z = self.grid_xyz(low_lims, up_lims, idxs=idxs,
                                               default_val=default_val)

        grid_x = grid_x.cpu().detach().numpy()
        grid_y = grid_y.cpu().detach().numpy()
        grid_z = grid_z.cpu().detach().numpy()
        axs.contour(grid_x, grid_y, grid_z, levels=30)

        kde_contour_plot_2d(axs, self.XA[:, idxs[0]], self.XA[:, idxs[1]],
                            weights=(self.weights["XA"].cpu().detach().numpy()
                                     if self.weights else None))
        kde_contour_plot_2d(axs, self.XB[:, idxs[0]], self.XB[:, idxs[1]],
                            weights=(self.weights["XB"].cpu().detach().numpy()
                                     if self.weights else None))

        axs.set_xlim([np.min(grid_x), np.max(grid_x)])
        axs.set_ylim([np.min(grid_y), np.max(grid_y)])
        axs.set_title("Coordinate contour")

    def plot_marginalised_dists(self, axs, flat_prior=False):
        XA_1d = self.net(self.XA_tnsr).squeeze().cpu().detach().numpy()
        XB_1d = self.net(self.XB_tnsr).squeeze().cpu().detach().numpy()
        X_prior_1d = (self.net(self.X_prior_tnsr).squeeze().cpu()
                                                 .detach().numpy())

        if flat_prior:
            XA_1d, XB_1d, X_prior_1d = self.flatten_prior(XA_1d, XB_1d,
                                                          X_prior_1d)

        kde_plot_1d(axs, XA_1d,
                    weights=(self.weights["XA"].cpu().detach().numpy() 
                             if self.weights else None))
        kde_plot_1d(axs, XB_1d,
                    weights=(self.weights["XB"].cpu().detach().numpy() 
                             if self.weights else None))
        kde_plot_1d(axs, X_prior_1d)
        axs.set_title("Marginalised 1d distribution"
                      f"{' - flat prior' if flat_prior else ''}")
    
    def flatten_prior(self, XA_1d, XB_1d, X_prior_1d):
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

        return XA_1d, XB_1d, X_prior_1d

    def get_bounds(self):
        bounds_A = get_limits(self.XA_tnsr)
        bounds_B = get_limits(self.XB_tnsr)
        bounds = torch.cat((bounds_A, bounds_B), dim=1)
        low_lims = bounds.min(1).values
        up_lims = bounds.max(1).values

        return low_lims, up_lims

    def grid_xyz(self, low_lims, up_lims, idxs, default_val=0):
        x = torch.linspace(low_lims[idxs[0]], up_lims[idxs[0]], 100,
                           device=self.device)
        y = torch.linspace(low_lims[idxs[1]], up_lims[idxs[1]], 100,
                           device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y)

        grid_xy = torch.cat((grid_x.unsqueeze(2), grid_y.unsqueeze(2)), dim=2)
        points_xy = grid_xy.view(-1, 2)

        X_prior_mean = self.X_prior_tnsr.mean(0)
        points = X_prior_mean.unsqueeze(0).repeat(points_xy.shape[0], 1)
        # points = torch.zeros((points_xy.shape[0], self.input_dim),
        #                      device=self.device) + default_val
        points[:, idxs[0]] = points_xy[:, 0]
        points[:, idxs[1]] = points_xy[:, 1]

        z = self.net(points)
        grid_z = z.view(100, 100)

        return grid_x, grid_y, grid_z
