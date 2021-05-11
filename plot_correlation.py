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
 params, param_means, norm_factors, param_stds) = planck_des_data(
     params=params, div_max=True, std=True
)
weights = {"XA": X0_weights, "XB": X1_weights, "X_prior": X_prior_weights}
rel_regions = [(param_means[i] - 10*param_stds[i], param_means[i] + 10*param_stds[i]) for i in range(6)]
param_means = torch.tensor(param_means).float().to(device)
norm_factors = torch.tensor(norm_factors).float().to(device)

X_prior_t = torch.tensor(X_prior).to(device).float()
fig, axs = plt.subplots(20, 6, figsize=(24, 80))

coeffs = [[] for _ in range(6)]

for j in range(20):
    tension_R = TensionNet1(6, hidden_size=4096)
    tension_R.load_state_dict(torch.load(f"plots/six_{j}.pt", map_location=device))
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
        x = torch.linspace(rel_regions[i][0], rel_regions[i][1], 1000)
        X_temp = X.clone().detach()
        X_temp[:, i] = x
        t = tension_R(X_temp).squeeze().detach().numpy()
        t = cdf_f(t)
        axs[j, i].plot(x * norm_factors[i], t)
        axs[j, i].set_ylim([0, 1])
        # coeff = np.corrcoef(x * norm_factors[i], t)[0][1]
        coeff, intercept = np.polyfit(x, t, 1)
        axs[j, i].set_title(round(coeff, 4))
        coeffs[i].append(abs(coeff))

coeffs_ave = [np.mean(corr) for corr in coeffs]
print(coeffs_ave)

        
plt.savefig("test.png", dpi=300)