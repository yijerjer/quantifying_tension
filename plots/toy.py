import os
# os.chdir("../")
import torch
from torch.optim import optimizer
import matplotlib.pyplot as plt
from matplotlib import rc
import torch.optim as optim
from np_utils import simple_data, curved_data, planck_des_data
from torch_utils import rotation_test, get_limits, TrainUtil
# from tension_net import TensionNet, TensionNet1, TensionNet2, TensionNet3
from tension_quantify import GaussianKDE, BayesFactorKDE, BayesFactor
# os.chdir("plots/")


rc('text', usetex=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X0, X1, X_prior = simple_data(dims=2)
X0_c, X1_c, X_prior_c = curved_data(dims=2)

simple_t = torch.load("toy_simple.pt")
curved_t = torch.load("toy_curved.pt")
optimizer = optim.Adam(simple_t.parameters(), lr=0.001)
criterion = BayesFactor()

simple_tu = TrainUtil(simple_t, optimizer, criterion, device)
simple_tu.XA = X0
simple_tu.XB = X1
simple_tu.X_prior = X_prior

simple_tu.XA_tnsr = torch.tensor(X0).to(device).float()
simple_tu.XB_tnsr = torch.tensor(X1).to(device).float()
simple_tu.X_prior_tnsr = torch.tensor(X_prior).to(device).float()

fig, axs = plt.subplots(figsize=(5, 5))
simple_tu.visualise_coordinate(axs)

plt.show()
