from anesthetic.plot import kde_plot_1d
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
from np_utils import simple_data, curved_data
from torch_utils import TrainUtil
from tension_net import TensionNet1
from tension_quantify import BayesFactorKDE, SuspiciousnessKDE, sigma_from_logS


rc('text', usetex=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
suss = SuspiciousnessKDE(device, n_points=500)

X0, X1, X_prior = simple_data(dims=2)
simple_t = TensionNet1(2, hidden_size=4096)
simple_t.load_state_dict(torch.load("plots/toy_simple.pt", map_location=device))
optimizer = optim.Adam(simple_t.parameters(), lr=0.001)
criterion = BayesFactorKDE(device)

simple_tu = TrainUtil(simple_t, optimizer, criterion, device)
simple_tu.XA = X0
simple_tu.XB = X1
simple_tu.X_prior = X_prior
simple_tu.weights = None

simple_tu.XA_tnsr = torch.tensor(X0).to(device).float()
simple_tu.XB_tnsr = torch.tensor(X1).to(device).float()
simple_tu.X_prior_tnsr = torch.tensor(X_prior).to(device).float()

# XA_1d = simple_tu.net(simple_tu.XA_tnsr)
# XB_1d = simple_tu.net(simple_tu.XB_tnsr)
# X_prior_1d = simple_tu.net(simple_tu.X_prior_tnsr)
# logR = criterion(XA_1d, XB_1d, X_prior_1d)
# logS = suss(XA_1d, XB_1d, X_prior_1d)
# sigma, p = sigma_from_logS((2, 0), (logS, 0))
# logR_1d = criterion(simple_tu.XA_tnsr[:, 0][None], simple_tu.XB_tnsr[:, 0][None], simple_tu.X_prior_tnsr[:, 0][None])
# logS_1d = suss(simple_tu.XA_tnsr[:, 0][None], simple_tu.XB_tnsr[:, 0][None], simple_tu.X_prior_tnsr[:, 0][None])
# sigma_1d, p_1d = sigma_from_logS((1, 0), (logS_1d, 0))
# print("".ljust(10), "1D".ljust(10), "2D".ljust(10))
# print("logR".ljust(10), str(round(logR, 2)).ljust(10),
#       str(round(logR_1d, 2)).ljust(10))
# print("logS".ljust(10), str(round(logS, 2)).ljust(10),
#       str(round(logS_1d, 2)).ljust(10))
# print("p".ljust(10), str(round(p[0], 2)).ljust(10),
#       str(round(p_1d[0], 2)).ljust(10))
# print("sgma".ljust(10), str(round(sigma[0], 2)).ljust(10),
#       str(round(sigma_1d[0], 2)).ljust(10))


X0_c, X1_c, X_prior_c = curved_data(dims=2, banana="more")
curved_t = TensionNet1(2, hidden_size=4096)
curved_t.load_state_dict(torch.load("plots/toy_curved.pt", map_location=device))
optimizer = optim.Adam(curved_t.parameters(), lr=0.001)
criterion = BayesFactorKDE(device)

curved_tu = TrainUtil(curved_t, optimizer, criterion, device)
curved_tu.XA = X0_c
curved_tu.XB = X1_c
curved_tu.X_prior = X_prior_c
curved_tu.weights = None

curved_tu.XA_tnsr = torch.tensor(X0_c).to(device).float()
curved_tu.XB_tnsr = torch.tensor(X1_c).to(device).float()
curved_tu.X_prior_tnsr = torch.tensor(X_prior_c).to(device).float()

# fig, axs = plt.subplots(2, 1, figsize=(4, 6))

# simple_tu.visualise_coordinate(axs[0], pad_div=10)
# curved_tu.visualise_coordinate(axs[1], pad_div=10)
# axs[0].set_title("")
# axs[1].set_title("")
# axs[0].text(-0.1, 1, r"$\rm{a)}$", transform=axs[0].transAxes, size=14)
# axs[1].text(-0.1, 1, r"$\rm{b)}$", transform=axs[1].transAxes, size=14)
# axs[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
# axs[0].tick_params(axis='y', which='both', left=False, labelleft=False)
# axs[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
# axs[1].tick_params(axis='y', which='both', left=False, labelleft=False)
# fig.tight_layout()

fig = plt.figure(figsize=(10, 4.5))
outer = gridspec.GridSpec(1, 15)

simple_gs = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer[:7], hspace=0.1, wspace=0.1)
curved_gs = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer[8:], hspace=0.1, wspace=0.1)

# smain_ax = outer.add_subplot(simple_gs[:-1, 1:])
smain_ax = plt.Subplot(fig, simple_gs[:-1, 1:])
sy_ax = plt.Subplot(fig, simple_gs[:-1, 0])
sx_ax = plt.Subplot(fig, simple_gs[-1, 1:])
smain_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
smain_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
sy_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
sy_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
sx_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
sx_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
fig.add_subplot(smain_ax)
fig.add_subplot(sy_ax)
fig.add_subplot(sx_ax)

simple_tu.visualise_coordinate(smain_ax, pad_div=10, norm_tension=True)
smain_ax.set_title("")
lims = smain_ax.viewLim.get_points()
xlims = lims[:, 0]
ylims = lims[:, 1]
kde_plot_1d(sy_ax, X0[:, 1], color="tab:orange")
kde_plot_1d(sy_ax, X1[:, 1], color="tab:green")
newx = sy_ax.lines[0].get_ydata()
newy = sy_ax.lines[0].get_xdata()
sy_ax.lines[0].set_xdata(newx)
sy_ax.lines[0].set_ydata(newy)
newx = sy_ax.lines[1].get_ydata()
newy = sy_ax.lines[1].get_xdata()
sy_ax.lines[1].set_xdata(newx)
sy_ax.lines[1].set_ydata(newy)
kde_plot_1d(sx_ax, X0[:, 0], color="tab:orange")
kde_plot_1d(sx_ax, X1[:, 0], color="tab:green")

sy_ax.set_xlim([0, 1.1])
sy_ax.set_ylim(ylims)
sx_ax.set_ylim([0, 1.1])
sx_ax.set_xlim(xlims)
sy_ax.invert_xaxis()
sx_ax.invert_yaxis()


cmain_ax = plt.Subplot(fig, curved_gs[:-1, 1:])
cy_ax = plt.Subplot(fig, curved_gs[:-1, 0])
cx_ax = plt.Subplot(fig, curved_gs[-1, 1:])
cmain_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
cmain_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
cy_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
cy_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
cx_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
cx_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
fig.add_subplot(cmain_ax)
fig.add_subplot(cy_ax)
fig.add_subplot(cx_ax)

curved_tu.visualise_coordinate(cmain_ax, pad_div=10, norm_tension=True)
cmain_ax.set_title("")
lims = cmain_ax.viewLim.get_points()
xlims = lims[:, 0]
ylims = lims[:, 1]
kde_plot_1d(cy_ax, X0_c[:, 1], color="tab:orange")
kde_plot_1d(cy_ax, X1_c[:, 1], color="tab:green")
newx = cy_ax.lines[0].get_ydata()
newy = cy_ax.lines[0].get_xdata()
cy_ax.lines[0].set_xdata(newx)
cy_ax.lines[0].set_ydata(newy)
newx = cy_ax.lines[1].get_ydata()
newy = cy_ax.lines[1].get_xdata()
cy_ax.lines[1].set_xdata(newx)
cy_ax.lines[1].set_ydata(newy)
kde_plot_1d(cx_ax, X0_c[:, 0], color="tab:orange")
kde_plot_1d(cx_ax, X1_c[:, 0], color="tab:green")

cy_ax.set_xlim([0, 1.1])
cy_ax.set_ylim(ylims)
cx_ax.set_ylim([0, 1.1])
cx_ax.set_xlim(xlims)
cy_ax.invert_xaxis()
cx_ax.invert_yaxis()

sy_ax.text(0, 1.05, r"$\rm{a)}$", transform=sy_ax.transAxes, size=16)
cy_ax.text(0, 1.05, r"$\rm{b)}$", transform=cy_ax.transAxes, size=16)

# plt.subplots_adjust(hspace=0.15)
# plt.show()
fig.tight_layout()
# plt.savefig("plots/toy_wide.png", dpi=300)