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

train_utils = []
for pair in param_pairs:
    net = TensionNet1(2, hidden_size=4096)
    net.load_state_dict(torch.load(f"plots/pair/{pair[0]}{pair[1]}_net.pt", map_location=device))
    criterion = BayesFactorKDE(device, n_points=500)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    util = TrainUtil(net, optimizer, criterion, device, 
                     data_labels=[r"$Planck$", r"\rm{DES}", r"\rm{Prior}"])

    util.XA = X0[:, pair]
    util.XB = X1[:, pair]
    util.X_prior = X_prior[:, pair]
    util.weights = {}
    util.weights["XA"] = torch.tensor(X0_weights).to(device).float()
    util.weights["XB"] = torch.tensor(X1_weights).to(device).float()
    util.weights["X_prior"] = torch.tensor(X_prior_weights).to(device).float()

    util.XA_tnsr = torch.tensor(util.XA).to(device).float()
    util.XB_tnsr = torch.tensor(util.XB).to(device).float()
    util.X_prior_tnsr = torch.tensor(util.X_prior).to(device).float()

    train_utils.append(util)


fig, axs = plt.subplots(5, 5, figsize=(10, 10), sharex='col', sharey='row')

for i in range(5):
    for j in range(5):
        if i <= j:
            pair = [i, j + 1]
            idx = param_pairs.index(pair)
            train_utils[idx].visualise_coordinate(
                axs[j, i], focus='both', pad_div=100, swap_order=True,
                param_means=param_means[pair],
                norm_factors=norm_factors[pair], norm_tension=True
            )
            axs[j, i].set_title("")

            if j != 4 and i != 0:
                axs[j, i].set_xlabel("")
                axs[j, i].set_ylabel("")
            elif j == 4 and i != 0:
                axs[j, i].set_ylabel("")
                axs[j, i].set_xlabel(param_labels[i])
            elif j != 4 and i == 0:
                axs[j, i].set_xlabel("")
                axs[j, i].set_ylabel(param_labels[j + 1])
            elif j == 4 and i == 0:
                axs[j, i].set_xlabel(param_labels[i])
                axs[j, i].set_ylabel(param_labels[j + 1])
        else:
            fig.delaxes(axs[j, i])

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1))
fig.tight_layout(rect=[0, 0, 1, 0.97])

plt.subplots_adjust(hspace=0.15, wspace=0.15)
fig.savefig("plots/pairs.png", dpi=300)