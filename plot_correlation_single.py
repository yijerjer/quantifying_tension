import numpy as np
import torch
from torch.functional import norm
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


rc('text', usetex=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = ["omegabh2", "omegam", "H0", "tau", "sigma8", "ns"]
param_labels = [r"$\Omega_b h^2$", r"$\Omega_m$", r"$H_0$", r"$\tau$", r"$\sigma_8$", r"$n_s$"]
param_pairs = []
for i in range(len(params)):
    for j in range(i + 1, len(params)):
        param_pairs.append([i, j])

(X0, X0_weights, X1, X1_weights, X_prior, X_prior_weights,
 params, param_means, norm_factors, param_stds) = planck_des_data(
     params=params, div_max=True, std=True
)
weights = {"XA": X0_weights, "XB": X1_weights, "X_prior": X_prior_weights}
avg0 = np.average(X0, axis=0, weights=X0_weights)
avg1 = np.average(X1, axis=0, weights=X1_weights)
# rel_regions = [(param_means[i] - 10*param_stds[i], param_means[i] + 10*param_stds[i]) for i in range(6)]
rel_regions = list(zip(avg0, avg1))
param_means = torch.tensor(param_means).float().to(device)
norm_factors = torch.tensor(norm_factors).float().to(device)

X_prior_t = torch.tensor(X_prior).to(device).float()
fig, axs = plt.subplots(3, 6, figsize=(16, 8), sharey='row')

coeffs = [[] for _ in range(6)]

j = 13
tension_R = TensionNet1(6, hidden_size=4096)
tension_R.load_state_dict(torch.load(f"plots/six_2/six_{j}.pt", map_location=device))
criterion = BayesFactorKDE(device)
optimizer = optim.Adam(tension_R.parameters(), lr=0.001)

X_prior_1d = tension_R(X_prior_t).squeeze().detach().numpy()
kde = gaussian_kde(X_prior_1d, weights=X_prior_weights)
pad = (np.max(X_prior_1d) - np.min(X_prior_1d))
y = np.linspace(np.min(X_prior_1d) - pad, np.max(X_prior_1d) + pad, 1000)
cdf = np.cumsum(kde(y))
cdf /= np.max(cdf)
cdf_f = interp1d(y, cdf)

X = param_means.repeat(1000, 1)

for i in range(6):
    x = torch.linspace(0, 1, 1000)
    X_temp = X.clone().detach()
    X_temp[:, i] = x
    t = tension_R(X_temp).squeeze().detach().numpy()
    t = cdf_f(t)
    axs[1, i].plot(x * norm_factors[i], t)
    axs[1, i].set_ylim([0, 1])
    axs[1, i].axvline(rel_regions[i][0] * norm_factors[i].item(), color="grey", lw=1)
    axs[1, i].axvline(rel_regions[i][1] * norm_factors[i].item(), color="grey", lw=1)
    axs[1, i].axvline((param_means[i] - 3 * param_stds[i]) * norm_factors[i], color='r', lw=1)
    axs[1, i].axvline((param_means[i] + 3 * param_stds[i]) * norm_factors[i], color='r', lw=1)
axs[1, 0].set_ylabel(r"$t$")

for i in range(6):
    x = torch.linspace(param_means[i] - 3 * param_stds[i], param_means[i] + 3 * param_stds[i], 1000)
    X_temp = X.clone().detach()
    X_temp[:, i] = x
    t = tension_R(X_temp).squeeze().detach().numpy()
    t = cdf_f(t)
    axs[2, i].plot(x * norm_factors[i], t)
    axs[0, i].set_xlabel(param_labels[i])
    # axs[2, i].set_ylim([0, 1])
axs[2, 0].set_ylabel(r"$t$")


for i in range(6):
    kde_plot_1d(axs[0, i], X1[:, i] * norm_factors[i].item(), weights=X1_weights, color="tab:orange")
    kde_plot_1d(axs[0, i], X0[:, i] * norm_factors[i].item(), weights=X0_weights, color="tab:green")
    axs[0, i].axvline(rel_regions[i][0] * norm_factors[i].item(), color="grey", lw=1)
    axs[0, i].axvline(rel_regions[i][1] * norm_factors[i].item(), color="grey", lw=1)
    axs[0, i].axvline((param_means[i] - 3 * param_stds[i]) * norm_factors[i], color='r', lw=1)
    axs[0, i].axvline((param_means[i] + 3 * param_stds[i]) * norm_factors[i], color='r', lw=1)


fig.suptitle("Varying a single parameter in 6-to-1 neural network.")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("test.png", dpi=300)