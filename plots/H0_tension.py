import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

planck = [67.4, 0.5, 0.5]
des = [67.4, 1.1, 1.2]
shoes = [74.0, 1.4, 1.4]
holicow = [73.3, 1.7, 1.8]
cchp = [69.6, 2.5, 2.5]

xs = [planck, des, cchp, holicow, shoes]
ys = [6, 4.5, 3, 1.5, 0]
labels = [r'$\textrm{Planck}$', r'\textrm{DES + BAO + BBN}', r'$\textrm{TRGB}$', r'\textrm{H0LiCOW}', r'$\textrm{SH0ES}$']
fig, axs = plt.subplots(figsize=(4, 3.5))

for i, x in enumerate(xs):
    axs.scatter(x[0], ys[i], label=labels[i])
    axs.plot([x[0] + x[1], x[0] - x[2]], [ys[i], ys[i]], linewidth='2')

    text = r"$%.1f^{+%.1f}_{-%.1f}$" % (x[0], x[1], x[2])
    axs.annotate(text, xy=(x[0], ys[i] + 0.3), ha="center", fontsize=13)
    axs.annotate(labels[i], xy=(x[0], ys[i] - 0.5), ha='center', fontsize=10, color='dimgrey')

axs.set_xlabel(r"$H_0 \rm{ \ / \ km \ s^{-1} \ Mpc^{-1}}$", fontsize=14)
axs.set_ylim([-1, 7])
axs.set_xlim([65, 76])
axs.tick_params(axis='y', which='both', left=False, labelleft=False)
plt.tight_layout()
fig.savefig('H0 tension.png', dpi=300)
