import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from anesthetic.plot import kde_plot_1d, kde_contour_plot_2d
from torch_utils import (get_limits, visualise_tension, visualise_coordinate,
                         plot_marginalised_dists)
from np_utils import simple_data, curved_data, uniform_prior_samples
from tension_net import TensionNet1
from tension_quantify import BayesFactor, SuspiciousnessKLDiv


def get_xy_points(XA_tensor, XB_tensor):
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

    return points_xy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X0, X1, X_prior = simple_data()
X0_tensor = torch.tensor(X0).float().to(device)
X1_tensor = torch.tensor(X1).float().to(device)
X_prior_tensor = torch.tensor(X_prior).float().to(device)

tension_R = TensionNet1(2).to(device)
criterion = BayesFactor(hist_type="gaussian", hist_param=1, n_dist_bins=100,
                        n_prior_bins=50).to(device)
optimizer = optim.SGD(tension_R.parameters(), lr=0.0005)

losses_R = []
points_xy = get_xy_points(X0_tensor, X1_tensor)
grid_xy = points_xy.view(100, 100, 2)
grid_x = grid_xy[:, :, 0].detach().numpy()
grid_y = grid_xy[:, :, 1].detach().numpy()
grids_z = np.empty((0, 100, 100))

iter = 100

for i in range(iter):
    z = tension_R(points_xy).view(100, 100)
    grids_z = np.concatenate((grids_z, np.array([z.detach().numpy()])))

    optimizer.zero_grad()
    X0_1d = tension_R(X0_tensor)
    X1_1d = tension_R(X1_tensor)
    X_prior_1d = tension_R(X_prior_tensor)

    loss = criterion(X0_1d, X1_1d, X_prior_1d)
    losses_R.append(loss.item())
    loss.backward()
    optimizer.step()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].set_xlim([0, len(losses_R)])
axs[1].set_xlim((points_xy[0][0], points_xy[-1][0]))
axs[1].set_ylim((points_xy[0][1], points_xy[-1][1]))

kde_contour_plot_2d(axs[1], X0[:, 0], X0[:, 1])
kde_contour_plot_2d(axs[1], X1[:, 0], X1[:, 1])

contour = axs[1].contour(grid_x, grid_y, grids_z[0], levels=30)
line, = axs[0].plot([], [])


def animate(i):
    global contour, line
    for c in contour.collections:
        c.remove()
    axs[0].plot(np.arange(len(losses_R[:i])), losses_R[:i], color='tab:blue')
    axs[0].set_xlim([0, iter])
    axs[0].set_ylim([-10, 2])
    contour = axs[1].contour(grid_x, grid_y, grids_z[i], levels=30)
    axs[1].set_xlim((points_xy[0][0], points_xy[-1][0]))
    axs[1].set_ylim((points_xy[0][1], points_xy[-1][1]))
    axs[1].set_title(f"{i}")
    return contour

anim = FuncAnimation(fig, animate, iter, repeat=False, interval=100)

plt.show()
