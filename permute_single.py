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

weights_t = {}
weights_t["XA"] = torch.tensor(weights["XA"]).to(device).float()
weights_t["XB"] = torch.tensor(weights["XB"]).to(device).float()
weights_t["X_prior"] = torch.tensor(weights["X_prior"]).to(device).float()
X0_t = torch.tensor(X0).to(device).float()
X1_t = torch.tensor(X1).to(device).float()
X_prior_t = torch.tensor(X_prior).to(device).float()

for j in range(20):
    tension_R = TensionNet1(6, hidden_size=4096)
    tension_R.load_state_dict(torch.load(f"plots/six_2/six_{j}.pt", map_location=device))
    criterion = BayesFactorKDE(device)
    optimizer = optim.Adam(tension_R.parameters(), lr=0.001)
    losses = []

    X0_1d = tension_R(X0_t)
    X1_1d = tension_R(X1_t)
    X_prior_1d = tension_R(X_prior_t)
    loss = criterion(X0_1d, X1_1d, X_prior_1d, weights=weights_t)
    losses.append(loss.item())

    for i in range(6):
        X0_r = X0_t.clone().detach()
        X1_r = X1_t.clone().detach()
        X_prior_r = X_prior_t.clone().detach()
        # X0_r[:, i] = X0_r[torch.randperm(X0_r.shape[0]), i]
        # X1_r[:, i] = X1_r[torch.randperm(X1_r.shape[0]), i]
        X0_r[:, i] = torch.rand(X0_r.shape[0])
        X1_r[:, i] = torch.rand(X1_r.shape[0])
        X_prior_r[:, i] = torch.rand(X_prior_r.shape[0])

        X0_1d = tension_R(X0_r)
        X1_1d = tension_R(X1_r)
        X_prior_1d = tension_R(X_prior_r)

        loss = criterion(X0_1d, X1_1d, X_prior_1d, weights=weights_t)
        losses.append(loss.item())
        del X0_r, X1_r

    print(losses)