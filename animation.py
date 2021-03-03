import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from np_utils import simple_data, curved_data
from anesthetic.plot import kde_contour_plot_2d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X0, X1, X_prior = simple_data()
data = np.load("suss_simple_exploding.npz")
losses = data['losses']
coords = data['coords']
dists = data['kdes']

grids_x = coords[:, 0, :, :]
grids_y = coords[:, 1, :, :]
grids_z = coords[:, 2, :, :]

dists_A = dists[:, 0, :, :]
dists_B = dists[:, 1, :, :]
dists_prior = dists[:, 2, :, :]
maxes_A = np.max(dists_A[:, :, 1], axis=1)
maxes_B = np.max(dists_B[:, :, 1], axis=1)
maxes_prior = np.max(dists_prior[:, :, 1], axis=1)
dists_A[:, :, 1] /= maxes_A[:, None]
dists_B[:, :, 1] /= maxes_B[:, None]
dists_prior[:, :, 1] /= maxes_prior[:, None]


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
kde_contour_plot_2d(axs[1], X0[:, 0], X0[:, 1])
kde_contour_plot_2d(axs[1], X1[:, 0], X1[:, 1])
contour = axs[1].contour(grids_x[0], grids_y[0], grids_z[0], levels=30)

line_A, = axs[2].plot(dists_A[0][:, 0], dists_A[0][:, 1])
line_B, = axs[2].plot(dists_B[0][:, 0], dists_B[0][:, 1])
line_prior, = axs[2].plot(dists_prior[0][:, 0], dists_prior[0][:, 1])
axs[2].set_xlim([np.min(dists[:, :, :, 0]), np.max(dists[:, :, :, 0])])

iter = len(losses)

def animate(i):
    global contour, line
    for c in contour.collections:
        c.remove()
    axs[0].plot(np.arange(len(losses[:i])), losses[:i], color='tab:blue')
    axs[0].set_xlim([0, iter])
    # axs[0].set_ylim([-10, 2])
    axs[0].set_title(f"Current loss = {round(losses[i], 4)}")

    contour = axs[1].contour(grids_x[0], grids_y[0], grids_z[i], levels=30)
    axs[1].set_xlim((np.min(grids_x), np.max(grids_x)))
    axs[1].set_ylim((np.min(grids_y), np.max(grids_y)))
    axs[1].set_title(f"{i}")

    line_A.set_data(dists_A[i][:, 0], dists_A[i][:, 1])
    line_B.set_data(dists_B[i][:, 0], dists_B[i][:, 1])
    line_prior.set_data(dists_prior[i][:, 0], dists_prior[i][:, 1])
    return contour

anim = FuncAnimation(fig, animate, iter, repeat=False, interval=20)
# anim.save("suss_simple_exploding.mp4")

plt.show()
