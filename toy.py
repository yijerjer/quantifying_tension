import os
# os.chdir("../")
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import rc
import torch.optim as optim
from np_utils import simple_data, curved_data, planck_des_data
from torch_utils import rotation_test, get_limits, TrainUtil
from tension_net import TensionNet, TensionNet1, TensionNet2, TensionNet3
from tension_quantify import GaussianKDE, BayesFactorKDE, BayesFactor
# os.chdir("plots/")


rc('text', usetex=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X0, X1, X_prior = simple_data(dims=2)
simple_t = TensionNet1(2, hidden_size=4096)
simple_t.load_state_dict(torch.load("plots/toy_simple.pt", map_location=device))
optimizer = optim.Adam(simple_t.parameters(), lr=0.001)
criterion = BayesFactor()

simple_tu = TrainUtil(simple_t, optimizer, criterion, device)
simple_tu.XA = X0
simple_tu.XB = X1
simple_tu.X_prior = X_prior
simple_tu.weights = None

simple_tu.XA_tnsr = torch.tensor(X0).to(device).float()
simple_tu.XB_tnsr = torch.tensor(X1).to(device).float()
simple_tu.X_prior_tnsr = torch.tensor(X_prior).to(device).float()

# fig, axs = plt.subplots(figsize=(5, 5))
# simple_tu.visualise_coordinate(axs)
# axs.set_title("")
# plt.show()


X0_c, X1_c, X_prior_c = curved_data(dims=2)
curved_t = TensionNet1(2, hidden_size=4096)
curved_t.load_state_dict(torch.load("plots/toy_curved.pt", map_location=device))
optimizer = optim.Adam(curved_t.parameters(), lr=0.001)
criterion = BayesFactor()

curved_tu = TrainUtil(curved_t, optimizer, criterion, device)
curved_tu.XA = X0_c
curved_tu.XB = X1_c
curved_tu.X_prior = X_prior_c
curved_tu.weights = None

curved_tu.XA_tnsr = torch.tensor(X0_c).to(device).float()
curved_tu.XB_tnsr = torch.tensor(X1_c).to(device).float()
curved_tu.X_prior_tnsr = torch.tensor(X_prior_c).to(device).float()

fig, axs = plt.subplots(2, 1, figsize=(4, 6))

simple_tu.visualise_coordinate(axs[0], pad_div=10)
curved_tu.visualise_coordinate(axs[1], pad_div=10)
axs[0].set_title("")
axs[1].set_title("")
axs[0].text(-0.1, 1, r"$\rm{a)}$", transform=axs[0].transAxes, size=14)
axs[1].text(-0.1, 1, r"$\rm{b)}$", transform=axs[1].transAxes, size=14)
axs[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
axs[0].tick_params(axis='y', which='both', left=False, labelleft=False)
axs[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
axs[1].tick_params(axis='y', which='both', left=False, labelleft=False)
fig.tight_layout()

plt.subplots_adjust(hspace=0.15)
plt.savefig("plots/toy.png", dpi=300)