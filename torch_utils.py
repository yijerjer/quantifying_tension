import copy
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
    def __init__(self, net, optimizer, criterion, device, animation=False,
                 data_labels=["A", "B", "Prior"]):
        self.net = net.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.losses = []
        self.animation = animation
        self.input_dim = self.net.input_size
        self.data_labels = data_labels
        if self.animation:
            self.coordinates = []
            self.kde_dists = []

    def train(self, XA, XB, X_prior, weights=None, n_iter=500,
              save_every=None, decrease_lr_at=None):
        self.XA = XA
        self.XB = XB
        self.X_prior = X_prior

        self.XA_tnsr = torch.tensor(XA).to(self.device).float()
        self.XB_tnsr = torch.tensor(XB).to(self.device).float()
        self.X_prior_tnsr = torch.tensor(X_prior).to(self.device).float()

        if weights is not None:
            self.weights = {}
            self.weights["XA"] = (torch.tensor(weights["XA"]).to(self.device)
                                  .float())
            self.weights["XB"] = (torch.tensor(weights["XB"]).to(self.device)
                                  .float())
            self.weights["X_prior"] = (torch.tensor(weights["X_prior"])
                                       .to(self.device).float())
        else:
            self.weights = None

        self.nets = []
        for i in range(n_iter):
            self.optimizer.zero_grad()
            XA_1d = self.net(self.XA_tnsr)
            XB_1d = self.net(self.XB_tnsr)
            X_prior_1d = self.net(self.X_prior_tnsr)

            if save_every is not None:
                if i % save_every == 0:
                    self.nets.append(copy.deepcopy(self.net))

            loss = self.criterion(XA_1d, XB_1d, X_prior_1d,
                                  weights=self.weights)

            if decrease_lr_at:
                if loss < decrease_lr_at[0]:
                    for g in self.optimizer.param_groups:
                        g["lr"] /= 10
                    decrease_lr_at.pop(0)
                        
            self.losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

        if save_every is not None:
            return self.losses, self.nets
        else:
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

    def visualise_tension(self, axs, idxs=(0, 1), focus='both', pad_div=100,
                          swap_order=False, param_means=None,
                          norm_factors=None):
        XA_1d = self.net(self.XA_tnsr)
        XB_1d = self.net(self.XB_tnsr)
        X_combine = torch.cat((XA_1d, XB_1d))

        low_lims, up_lims = self.get_bounds(focus=focus, pad_div=pad_div)

        def likelihood_f(z, X_1d, cov=torch.tensor(1).float().to(self.device)):
            normal = dists.Normal(z.float(), cov)
            return normal.log_prob(X_1d).sum(0)

        grid_x, grid_y, grid_z = self.grid_xyz(low_lims, up_lims, idxs=idxs,
                                               param_means=param_means)
        if norm_factors is not None:
            grid_x = grid_x.clone() * norm_factors[idxs[0]]
            grid_y = grid_y.clone() * norm_factors[idxs[1]]

        z = grid_z.view(-1)
        z_llhd = likelihood_f(z, X_combine)
        grid_z_llhd = z_llhd.view(100, 100)

        grid_x = grid_x.cpu().detach().numpy()
        grid_y = grid_y.cpu().detach().numpy()
        grid_z_llhd = grid_z_llhd.cpu().detach().numpy()
        axs.contour(grid_x, grid_y, grid_z_llhd, levels=30)

        first = self.XB.copy() if swap_order else self.XA.copy()
        second = self.XA.copy() if swap_order else self.XB.copy()
        prior = self.X_prior.copy()
        if norm_factors is not None:
            np_norm_factors = norm_factors.cpu().detach().numpy()
            first *= np_norm_factors
            second *= np_norm_factors
            prior *= np_norm_factors
        if swap_order:
            first_w = (self.weights["XB"].cpu().detach().numpy()
                       if self.weights else None)
            second_w = (self.weights["XA"].cpu().detach().numpy()
                        if self.weights else None)
        else:
            first_w = (self.weights["XA"].cpu().detach().numpy()
                       if self.weights else None)
            second_w = (self.weights["XB"].cpu().detach().numpy()
                        if self.weights else None)
        kde_contour_plot_2d(
            axs, prior[:, idxs[0]], prior[:, idxs[1]],
            color='tab:blue', alpha=0.2, label=self.data_labels[2],
            weights=(self.weights["X_prior"].cpu().detach().numpy()
                     if self.weights else None)
        )
        kde_contour_plot_2d(axs, first[:, idxs[0]], first[:, idxs[1]],
                            color='tab:orange', weights=first_w,
                            label=self.data_labels[1 if swap_order else 0])
        kde_contour_plot_2d(axs, second[:, idxs[0]], second[:, idxs[1]],
                            color='tab:green', weights=second_w,
                            label=self.data_labels[0 if swap_order else 1])

        axs.set_xlim([np.min(grid_x), np.max(grid_x)])
        axs.set_ylim([np.min(grid_y), np.max(grid_y)])
        axs.set_title("Likelihood contour")

    def visualise_coordinate(self, axs, idxs=(0, 1), focus='both', pad_div=100,
                             swap_order=False, param_means=None,
                             norm_factors=None, norm_tension=False):
        low_lims, up_lims = self.get_bounds(focus=focus, pad_div=pad_div)

        grid_x, grid_y, grid_z = self.grid_xyz(low_lims, up_lims, idxs=idxs,
                                               param_means=param_means)
        if norm_factors is not None:
            grid_x = grid_x.clone() * norm_factors[idxs[0]]
            grid_y = grid_y.clone() * norm_factors[idxs[1]]

        if norm_tension:
            X_prior_1d = (self.net(self.X_prior_tnsr).squeeze().cpu()
                          .detach().numpy())
            abs_range = np.max(X_prior_1d) - np.min(X_prior_1d)
            t_range = [np.min(X_prior_1d) - abs_range,
                       np.max(X_prior_1d) + abs_range]
            norm_tension_f = self.flatten_prior_f(X_prior_1d, t_range)

        grid_x = grid_x.cpu().detach().numpy()
        grid_y = grid_y.cpu().detach().numpy()
        grid_z = grid_z.cpu().detach().numpy()
        if norm_tension:
            grid_z = norm_tension_f(grid_z)
        axs.contour(grid_x, grid_y, grid_z, linewidths=1, levels=15)

        first = self.XB.copy() if swap_order else self.XA.copy()
        second = self.XA.copy() if swap_order else self.XB.copy()
        prior = self.X_prior.copy()
        if norm_factors is not None:
            np_norm_factors = norm_factors.cpu().detach().numpy()
            first *= np_norm_factors
            second *= np_norm_factors
            prior *= np_norm_factors
        if swap_order:
            first_w = (self.weights["XB"].cpu().detach().numpy()
                       if self.weights else None)
            second_w = (self.weights["XA"].cpu().detach().numpy()
                        if self.weights else None)
        else:
            first_w = (self.weights["XA"].cpu().detach().numpy()
                       if self.weights else None)
            second_w = (self.weights["XB"].cpu().detach().numpy()
                        if self.weights else None)
        kde_contour_plot_2d(
            axs, prior[:, idxs[0]], prior[:, idxs[1]],
            color='tab:blue', alpha=0.2, label=self.data_labels[2],
            weights=(self.weights["X_prior"].cpu().detach().numpy()
                     if self.weights else None)
        )
        kde_contour_plot_2d(axs, first[:, idxs[0]], first[:, idxs[1]],
                            color='tab:orange', weights=first_w,
                            label=self.data_labels[1 if swap_order else 0])
        kde_contour_plot_2d(axs, second[:, idxs[0]], second[:, idxs[1]],
                            color='tab:green', weights=second_w,
                            label=self.data_labels[0 if swap_order else 1])

        axs.set_xlim([np.min(grid_x), np.max(grid_x)])
        axs.set_ylim([np.min(grid_y), np.max(grid_y)])
        axs.set_title("Coordinate contour")

    def visualise_coordinates_all(self, fig, axs, only_idxs=None,
                                  param_names=None, sync_levels=False,
                                  tension_as_param=False, focus='both',
                                  pad_div=100, swap_order=False,
                                  param_means=None, norm_factors=None,
                                  norm_tension=False, no_contour=False,
                                  plot_midpoint=False):
        if only_idxs is None:
            plot_size = self.input_dim
            only_idxs = np.arange(plot_size)
        else:
            plot_size = len(only_idxs)

        first = self.XB.copy() if swap_order else self.XA.copy()
        second = self.XA.copy() if swap_order else self.XB.copy()
        prior = self.X_prior.copy()
        if norm_factors is not None:
            np_norm_factors = norm_factors.cpu().detach().numpy()
            first *= np_norm_factors
            second *= np_norm_factors
            prior *= np_norm_factors
        if swap_order:
            first_w = (self.weights["XB"].cpu().detach().numpy()
                       if self.weights else None)
            second_w = (self.weights["XA"].cpu().detach().numpy()
                        if self.weights else None)
        else:
            first_w = (self.weights["XA"].cpu().detach().numpy()
                       if self.weights else None)
            second_w = (self.weights["XB"].cpu().detach().numpy()
                        if self.weights else None)
        prior_w = (self.weights["X_prior"].cpu().detach().numpy()
                   if self.weights else None)
        
        X_prior_1d = (self.net(self.X_prior_tnsr).squeeze().cpu()
                      .detach().numpy())
        
        XA_1d = self.net(self.XA_tnsr).squeeze().cpu().detach().numpy()
        XB_1d = self.net(self.XB_tnsr).squeeze().cpu().detach().numpy()
        first_1d = XB_1d if swap_order else XA_1d
        second_1d = XA_1d if swap_order else XB_1d

        if norm_tension:
            abs_range = np.max(X_prior_1d) - np.min(X_prior_1d)
            t_range = [np.min(X_prior_1d) - abs_range,
                       np.max(X_prior_1d) + abs_range]

            norm_tension_f = self.flatten_prior_f(X_prior_1d, t_range)
            first_1d = norm_tension_f(first_1d)
            second_1d = norm_tension_f(second_1d)
            X_prior_1d = norm_tension_f(X_prior_1d)

        if plot_midpoint:
            first_mean = np.average(first_1d, weights=first_w)
            second_mean = np.average(second_1d, weights=second_w)
            mid = (second_mean + first_mean) / 2

        all_grids = [[] for i in range(plot_size)]
        min_z = np.inf
        max_z = -np.inf
        for i in range(plot_size - 1):
            for j in range(plot_size - 1):
                ieff = i + 1
                jeff = j
                if ieff > jeff:
                    low_lims, up_lims = self.get_bounds(
                        focus=focus, pad_div=pad_div
                    )

                    grid_x, grid_y, grid_z = self.grid_xyz(
                        low_lims, up_lims, (only_idxs[jeff], only_idxs[ieff]),
                        param_means=param_means
                    )

                    if norm_factors is not None:
                        grid_x = grid_x.clone() * norm_factors[only_idxs[jeff]]
                        grid_y = grid_y.clone() * norm_factors[only_idxs[ieff]]

                    grid_x = grid_x.cpu().detach().numpy()
                    grid_y = grid_y.cpu().detach().numpy()
                    grid_z = grid_z.cpu().detach().numpy()
                    if norm_tension:
                        print(ieff, jeff, np.max(grid_z), np.min(grid_z))
                        grid_z = norm_tension_f(grid_z)

                    all_grids[ieff].append((grid_x, grid_y, grid_z))
                    min_z = min(np.min(grid_z), min_z)
                    max_z = max(np.max(grid_z), max_z)

        if norm_tension:
            levels = np.linspace(0, 1, 21)
        else:
            levels = np.linspace(min_z, max_z, 20)
        for i in range(plot_size - 1):
            for j in range(plot_size - 1):
                ieff = i + 1
                jeff = j
                if i >= j:
                    grid_x, grid_y, grid_z = all_grids[ieff][jeff]
                    if not no_contour:
                        cntr = axs[i, j].contour(
                            grid_x, grid_y, grid_z, linewidths=1,
                            levels=(levels if sync_levels else 20)
                        )
                        if plot_midpoint:
                            axs[i, j].contour(
                                grid_x, grid_y, grid_z, linewidths=1, 
                                levels=[mid], color='k', linewidth=2
                            )
                    kde_contour_plot_2d(
                        axs[i, j], prior[:, only_idxs[jeff]],
                        prior[:, only_idxs[ieff]], weights=prior_w,
                        color='tab:blue', alpha=0.2, label=self.data_labels[2]
                    )
                    kde_contour_plot_2d(
                        axs[i, j], first[:, only_idxs[jeff]],
                        first[:, only_idxs[ieff]], color='tab:orange',
                        weights=first_w,
                        label=self.data_labels[1 if swap_order else 0]
                    )
                    kde_contour_plot_2d(
                        axs[i, j], second[:, only_idxs[jeff]],
                        second[:, only_idxs[ieff]], color='tab:green',
                        weights=second_w,
                        label=self.data_labels[0 if swap_order else 1]
                    )

                    axs[i, j].set_xlim([grid_x.min(), grid_x.max()])
                    axs[i, j].set_ylim([grid_y.min(), grid_y.max()])
                    if param_names is not None:
                        axs[i, j].set_xlabel(param_names[only_idxs[jeff]])
                        axs[i, j].set_ylabel(param_names[only_idxs[ieff]])
                    else:
                        axs[i, j].set_xlabel(only_idxs[jeff])
                        axs[i, j].set_ylabel(only_idxs[ieff])
                else:
                    fig.delaxes(axs[i, j])

        if tension_as_param:
            for i in range(self.input_dim):
                final_row = self.input_dim - 1

                kde_contour_plot_2d(
                    axs[final_row, i], prior[:, i],
                    X_prior_1d, weights=prior_w, color='tab:blue',
                    alpha=0.2, label=self.data_labels[2]
                )
                kde_contour_plot_2d(
                    axs[final_row, i], first[:, i],
                    first_1d, weights=first_w, color='tab:orange',
                    label=self.data_labels[1 if swap_order else 0]
                )
                kde_contour_plot_2d(
                    axs[final_row, i], second[:, i],
                    second_1d, weights=second_w, color='tab:green',
                    label=self.data_labels[0 if swap_order else 1]
                )

                if i != (final_row):
                    fig.delaxes(axs[i, final_row])
                    grid_x, _, _ = all_grids[-1][i]
                    axs[final_row, i].set_xlim([grid_x.min(), grid_x.max()])

                axs[final_row, i].set_xlabel(param_names[only_idxs[i]])
                axs[final_row, i].set_ylabel("tension")

        if not no_contour:
            cbar_ax = fig.add_axes([0.90, 0.65, 0.03, 0.30])
            fig.colorbar(cntr, cax=cbar_ax)
        fig.tight_layout()

    def plot_marginalised_dists(self, axs, flat_prior=False, swap_order=False):
        XA_1d = self.net(self.XA_tnsr).squeeze().cpu().detach().numpy()
        XB_1d = self.net(self.XB_tnsr).squeeze().cpu().detach().numpy()
        X_prior_1d = (self.net(self.X_prior_tnsr).squeeze().cpu()
                      .detach().numpy())

        if flat_prior:
            XA_1d, XB_1d, X_prior_1d = self.flatten_prior(XA_1d, XB_1d,
                                                          X_prior_1d)

        first = XB_1d if swap_order else XA_1d
        second = XA_1d if swap_order else XB_1d
        if swap_order:
            first_w = (self.weights["XB"].cpu().detach().numpy()
                       if self.weights else None)
            second_w = (self.weights["XA"].cpu().detach().numpy()
                        if self.weights else None)
        else:
            first_w = (self.weights["XA"].cpu().detach().numpy()
                       if self.weights else None)
            second_w = (self.weights["XB"].cpu().detach().numpy()
                        if self.weights else None)

        prior_weights = (self.weights["X_prior"].cpu().detach().numpy()
                         if self.weights else None)
        kde_plot_1d(axs, X_prior_1d, color='tab:blue', weights=prior_weights,
                    label=self.data_labels[2])
        kde_plot_1d(axs, first, color='tab:orange', weights=first_w,
                    label=self.data_labels[1 if swap_order else 0])
        kde_plot_1d(axs, second, color='tab:green', weights=second_w,
                    label=self.data_labels[0 if swap_order else 1])
        axs.set_title("Marginalised 1d distribution"
                      f"{' - flat prior' if flat_prior else ''}")
    
    def flatten_prior_f(self, X_prior_1d, range):
        kde = gaussian_kde(
            X_prior_1d,
            weights=(self.weights["X_prior"].cpu().detach().numpy()
                     if self.weights else None)
        )
        pad = (range[1] - range[0]) / 100
        x = np.linspace(range[0] - pad, range[1] + pad, 1000)
        pdf = kde(x)
        cdf = np.cumsum(pdf)
        cdf /= np.max(cdf)
        cdf_f = interp1d(x, cdf)

        return cdf_f

    def flatten_prior(self, XA_1d, XB_1d, X_prior_1d):
        kde = gaussian_kde(
            X_prior_1d,
            weights=(self.weights["X_prior"].cpu().detach().numpy()
                     if self.weights else None)
        )
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

    def get_bounds(self, no_prior=True, focus='both', pad_div=100):
        if focus == 'both':
            X_combine = torch.cat((self.XA_tnsr, self.XB_tnsr), dim=0)
        elif focus == "A":
            X_combine = self.XA_tnsr
        elif focus == "B":
            X_combine = self.XB_tnsr
        else:
            raise ValueError("focus attribute must be either 'both', 'A', or 'B'.")

        if no_prior:
            bounds = get_limits(X_combine, pad_div=pad_div)
        else:
            bounds = get_limits(self.X_prior_tnsr, pad_div=pad_div)
        low_lims = bounds[:, 0]
        up_lims = bounds[:, 1]

        return low_lims, up_lims

    def grid_xyz(self, low_lims, up_lims, idxs, param_means=None):
        x = torch.linspace(low_lims[idxs[0]], up_lims[idxs[0]], 100,
                           device=self.device)
        y = torch.linspace(low_lims[idxs[1]], up_lims[idxs[1]], 100,
                           device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y)

        grid_xy = torch.cat((grid_x.unsqueeze(2), grid_y.unsqueeze(2)), dim=2)
        points_xy = grid_xy.view(-1, 2)

        if param_means is not None:
            X_prior_mean = param_means
        elif self.weights is None:
            X_prior_mean = self.X_prior_tnsr.mean(0)
        else:
            prior_weights = self.weights["X_prior"]
            X_prior_mean = self.X_prior_tnsr * prior_weights[:, None]
            X_prior_mean = X_prior_mean.sum(dim=0) / prior_weights.sum()

        points = X_prior_mean.repeat(points_xy.shape[0], 1)
        points[:, idxs[0]] = points_xy[:, 0]
        points[:, idxs[1]] = points_xy[:, 1]

        z = self.net(points)
        grid_z = z.view(100, 100)

        return grid_x, grid_y, grid_z
