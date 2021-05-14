import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from np_utils import planck_des_data
from tension_net import TensionNet1
from tension_quantify import (BayesFactorKDE, SuspiciousnessKDE,
                              sigma_from_logS)
from torch_utils import TrainUtil


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the DES and Planck data points from nested sampling chains
params = ["omegabh2", "omegam", "H0", "tau", "sigma8", "ns"]
param_labels = [r"$\Omega_b h^2$", r"$\Omega_m$", r"$H_0$",
                r"$\tau$", r"$\sigma_8$", r"$n_s$"]
(X0, X0_weights, X1, X1_weights, X_prior, X_prior_weights,
 params, param_means, norm_factors, param_stds) = planck_des_data(
     params=params, div_max=True, std=True
)
weights = {"XA": X0_weights, "XB": X1_weights, "X_prior": X_prior_weights}
param_means = torch.tensor(param_means).float().to(device)
norm_factors = torch.tensor(norm_factors).float().to(device)

# initialise training, and begin training
start = time.time()
tension_R = TensionNet1(6, hidden_size=4096)
criterion = BayesFactorKDE(device, n_points=500)
optimizer = optim.Adam(tension_R.parameters(), lr=0.0001)

train_util = TrainUtil(tension_R, optimizer, criterion, device)
losses = train_util.train(X0, X1, X_prior, weights=weights,
                          n_iter=500, decrease_lr_at=[-8])

# Calculate the sigma using Suspiciousness statistic
XA_1d = train_util.net(train_util.XA_tnsr)
XB_1d = train_util.net(train_util.XB_tnsr)
X_prior_1d = train_util.net(train_util.X_prior_tnsr)
suss = SuspiciousnessKDE(device, n_points=500)
logS = suss(XA_1d, XB_1d, X_prior_1d, weights=train_util.weights)
d = (4, 0)
sigma, p = sigma_from_logS(d, (logS, 0))

print("Training time taken: ", time.time() - start)
print("log R: ", losses[-1])
print("log S: ", logS)
print("sigma: ", sigma[0])

# Plot loss vs epoch, 2d contour plot of (0, 1) parameters,
# and marginalised posterior in tension coordinate
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
train_util.plot_loss(axs[0])
axs[0].set_title("Loss")
train_util.visualise_coordinate(
    axs[1], norm_tension=True, idxs=(0, 1), param_means=param_means,
    norm_factors=norm_factors, swap_order=True
)
train_util.plot_marginalised_dists(axs[3], flat_prior=True)
plt.show()

# Plot 2d contour plots of all parameter pairs
fig, axs = plt.subplots(6, 6, figsize=(10, 10), sharex='col', sharey='row')
train_util.visualise_coordinates_all(
    fig, axs, param_names=params,
    sync_levels=True, tension_as_param=True, focus='both', pad_div=100,
    swap_order=True, param_means=param_means, norm_factors=norm_factors,
    norm_tension=True
)

for i in range(6):
    for j in range(6):
        if i != 5 and j != 0:
            axs[i, j].set_xlabel("")
            axs[i, j].set_ylabel("")
        elif i == 5 and j != 0:
            axs[i, j].set_ylabel("")
            axs[i, j].set_xlabel(param_labels[j])
        elif i != 5 and j == 0:
            axs[i, j].set_xlabel("")
            axs[i, j].set_ylabel(param_labels[i + 1])
        elif i == 5 and j == 0:
            axs[i, j].set_xlabel(param_labels[j])
            axs[i, j].set_ylabel(r"$t$")

        if i == 5:
            axs[i, j].set_ylim([0, 1])

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center",
           ncol=3, bbox_to_anchor=(0.5, 1))
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(hspace=0.15, wspace=0.15)
plt.show()
