import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from np_utils import simple_data, curved_data, planck_des_data
from torch_utils import rotation_test, get_limits, TrainUtil
from tension_net import TensionNet, TensionNet1, TensionNet2, TensionNet3
from tension_quantify import GaussianKDE, BayesFactorKDE, BayesFactor
from anesthetic.plot import kde_plot_1d


def flatten_prior(XA_1d, XB_1d, X_prior_1d, weights=None):
    kde = gaussian_kde(
        X_prior_1d,
        weights=(weights["X_prior"].cpu().detach().numpy()
                 if weights else None)
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


rc('text', usetex=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = ["omegabh2", "omegam", "H0", "tau", "sigma8", "ns"]
param_labels = [r"$\Omega_b h^2$", r"$\Omega_m$", r"$H_0$", r"$\tau$", r"$\sigma_8$", r"$n_s$"]
param_pairs = []
for i in range(len(params)):
    for j in range(i + 1, len(params)):
        param_pairs.append([i, j])

(X0, X0_weights, X1, X1_weights, X_prior, X_prior_weights,
 params, param_means, norm_factors) = planck_des_data(
     params=params, div_max=True
)
weights = {"XA": X0_weights, "XB": X1_weights, "X_prior": X_prior_weights}
param_means = torch.tensor(param_means).float().to(device)
norm_factors = torch.tensor(norm_factors).float().to(device)

tension_R = TensionNet1(6, hidden_size=4096)
tension_R.load_state_dict(torch.load("plots/six_2/six_13.pt", map_location=device))
criterion = BayesFactorKDE(device)
optimizer = optim.Adam(tension_R.parameters(), lr=0.001)

train_util_R = TrainUtil(tension_R, optimizer, criterion, device, 
                         data_labels=[r"$Planck$", r"\rm{DES}", r"\rm{Prior}"])
train_util_R.losses = np.loadtxt("plots/six_2/six_13_loss.csv", delimiter=",")

train_util_R.XA = X0
train_util_R.XB = X1
train_util_R.X_prior = X_prior
train_util_R.weights = {}
train_util_R.weights["XA"] = torch.tensor(X0_weights).to(device).float()
train_util_R.weights["XB"] = torch.tensor(X1_weights).to(device).float()
train_util_R.weights["X_prior"] = torch.tensor(X_prior_weights).to(device).float()

train_util_R.XA_tnsr = torch.tensor(X0).to(device).float()
train_util_R.XB_tnsr = torch.tensor(X1).to(device).float()
train_util_R.X_prior_tnsr = torch.tensor(X_prior).to(device).float()

XA_1d = tension_R(train_util_R.XA_tnsr)
XB_1d = tension_R(train_util_R.XB_tnsr)
X_prior_1d = tension_R(train_util_R.X_prior_tnsr)
logR = criterion(XA_1d, XB_1d, X_prior_1d, weights=train_util_R.weights)
print("log Bayes Factor: ", logR)

fig, axs = plt.subplots(6, 6, figsize=(10, 10), sharex='col', sharey='row')
train_util_R.visualise_coordinates_all(
    fig, axs, param_names=params,
    sync_levels=True, tension_as_param=True, focus='both', pad_div=100,
    swap_order=True, param_means=param_means, norm_factors=norm_factors,
    norm_tension=True, plot_midpoint=True
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
            # axs[i, j].set_ylim([0.725, 0.825])
            axs[i, j].set_ylim([0, 1])

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1))
fig.tight_layout(rect=[0, 0, 1, 0.97])

plt.subplots_adjust(hspace=0.15, wspace=0.15)
# plt.show()
plt.savefig("plots/six.png", dpi=300)


# fig, axs = plt.subplots(2, 1, figsize=(4, 6))
# train_util_R.plot_marginalised_dists(axs[0], swap_order=True)
# train_util_R.plot_marginalised_dists(axs[1], flat_prior=True, swap_order=True)

# axs[0].set_title("")
# axs[1].set_title("")
# axs[0].text(-0.17, 1, r"$\rm{a)}$", transform=axs[0].transAxes, size=14)
# axs[1].text(-0.17, 1, r"$\rm{b)}$", transform=axs[1].transAxes, size=14)
# handles, labels = axs[1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.56, 1))
# fig.tight_layout(rect=[0, 0, 1, 0.97])


# plt.subplots_adjust(hspace=0.2)
# # plt.show()
# plt.savefig("plots/six_1d.png", dpi=300)


# Approximating distribution as Gaussian and using Gaussian tension

# X0_mean = (X0 * X0_weights[:, None]).sum(axis=0) / X0_weights.sum()
# X1_mean = (X1 * X1_weights[:, None]).sum(axis=0) / X1_weights.sum()
# X0_cov = np.cov(X0.T, aweights=X0_weights)
# X1_cov = np.cov(X1.T, aweights=X1_weights)

# mean_diff = X0_mean - X1_mean
# cov_sum = X0_cov + X1_cov

# X0_t = np.matmul(mean_diff, np.matmul(np.linalg.inv(cov_sum), X0.T))
# X1_t = np.matmul(mean_diff, np.matmul(np.linalg.inv(cov_sum), X1.T))
# X_prior_t = np.matmul(mean_diff, np.matmul(np.linalg.inv(cov_sum), X_prior.T))

# X0_t = torch.tensor(X0_t[:, None]).to(device)
# X1_t = torch.tensor(X1_t[:, None]).to(device)
# X_prior_t = torch.tensor(X_prior_t[:, None]).to(device)
# weights_t = {}
# weights_t["XA"] = torch.tensor(weights["XA"]).to(device)
# weights_t["XB"] = torch.tensor(weights["XB"]).to(device)
# weights_t["X_prior"] = torch.tensor(weights["X_prior"]).to(device)

# bf = BayesFactorKDE(device, n_points=500)
# logR = bf(X0_t, X1_t, X_prior_t, weights=weights_t)
# print("Assuming Gaussian distributions, logR: ", logR.item())

# X0_t, X1_t, X_prior_t = flatten_prior(
#     X0_t.squeeze().cpu().detach().numpy(),
#     X1_t.squeeze().cpu().detach().numpy(),
#     X_prior_t.squeeze().cpu().detach().numpy(),
#     weights=weights_t
# )

# fig, axs = plt.subplots(2, 1, figsize=(4, 6))
# train_util_R.plot_marginalised_dists(axs[1], flat_prior=True, swap_order=True)
# kde_plot_1d(axs[0], X_prior_t, weights=X_prior_weights)
# kde_plot_1d(axs[0], X1_t, weights=X1_weights)
# kde_plot_1d(axs[0], X0_t, weights=X0_weights)


# axs[0].set_xlabel(r"$t$")
# axs[1].set_xlabel(r"$t$")
# axs[0].set_title(r"\textrm{Gaussian Assumption}")
# axs[1].set_title(r"\textrm{Neural Network Approach}")
# axs[0].text(-0.1, 1, r"$\rm{a)}$", transform=axs[0].transAxes, size=14)
# axs[1].text(-0.1, 1, r"$\rm{b)}$", transform=axs[1].transAxes, size=14)
# axs[0].tick_params(axis='y', which='both', left=False, labelleft=False)
# axs[1].tick_params(axis='y', which='both', left=False, labelleft=False)

# handles, labels = axs[1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.56, 1))
# fig.tight_layout(rect=[0, 0, 1, 0.95])


# plt.subplots_adjust(hspace=0.35)
# # plt.show()
# plt.savefig("plots/six_1d_compare.png", dpi=300)