import os
import numpy
import scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from anesthetic import NestedSamples, make_2d_axes


def p(logS, d):
    return scipy.stats.chi2.sf(d[d>0]-2*logS[d>0], d[d>0])


def sigma(p):
    return numpy.sqrt(2)*scipy.special.erfcinv(p)


def ms(x):
    return x.mean(), x.std()


numpy.random.seed(0)

plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['text.usetex'] = True
serif = True
if serif:
    plt.rcParams["font.family"] = "serif"
else:
    plt.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = "cm"


labels = {
        'planck': r'Planck',
        'BAO': r'BAO',
        'SH0ES': r'S$H_0$ES',
        'lensing': r'lensing',
        'planck_lensing': r'Planck+lensing',
        'planck_BAO': r'Planck+BAO',
        'lensing_BAO': r'lensing+BAO',
        'planck_SH0ES': r'Planck+S$H_0$ES',
        'lensing_SH0ES': r'lensing+S$H_0$ES',
        'planck_lensing_BAO': r'Planck+lensing+BAO',
        'planck_lensing_SH0ES': r'Planck+lensing+S$H_0$ES',
        }

samples = {'lcdm': {}, 'klcdm': {}}
stats = {'lcdm': {}, 'klcdm': {}}
for model in tqdm(samples):
    for data, label in tqdm(labels.items()):
        root = os.path.join(model, 'chains', data)
        samples[model][data] = NestedSamples(root=root, label=label)
        stats[model][data] = samples[model][data].ns_output(1000)


# Plotting tensions.pdf
fig, ax = plt.subplots()
fig.set_size_inches(7, 3)
datasets = [
            ('planck', 'lensing', 'BAO'),
            ('planck_BAO', 'lensing'),
            ('planck', 'lensing_BAO'),
            ('planck_lensing', 'BAO'),
            (),
            ('planck', 'lensing', 'SH0ES'),
            ('planck_SH0ES', 'lensing'),
            ('planck', 'lensing_SH0ES'),
            ('planck_lensing', 'SH0ES'),
            (),
            ('lensing', 'BAO'),
            ('lensing', 'SH0ES'),
            ('planck', 'lensing'),
            ('planck', 'BAO'),
            ('planck', 'SH0ES'),
            ]
sigmas = {'lcdm': {}, 'klcdm': {}}
for y, data in enumerate(datasets):
    for model, color, label in [('lcdm', '#ff7f0e', r'$\Lambda$CDM'),
                                ('klcdm', '#1f77b4', r'$K\Lambda$CDM')]:
        if len(data) == 2:
            A, B = data
            if (A, B) == ('planck_BAO', 'lensing'):
                AB = 'planck_lensing_BAO'
            elif (A, B) == ('planck_SH0ES', 'lensing'):
                AB = 'planck_lensing_SH0ES'
            else:
                AB = A + '_' + B
            lZ = (stats[model][AB].logZ
                  - stats[model][A].logZ
                  - stats[model][B].logZ)
            lI = (stats[model][A].D
                  + stats[model][B].D
                  - stats[model][AB].D)
            if 'planck' not in A:
                BMD = (stats[model][A].d + stats[model][B].d
                       - stats[model][AB].d)
            else:
                BMD = stats[model][B].d

        elif len(data) == 3:
            A, B, C = data
            ABC = A + '_' + B + '_' + C
            lZ = (stats[model][ABC].logZ
                  - stats[model][A].logZ
                  - stats[model][B].logZ
                  - stats[model][C].logZ)
            lI = (stats[model][A].D
                  + stats[model][B].D
                  + stats[model][C].D
                  - stats[model][ABC].D)
            BMD = stats[model][B].d + stats[model][C].d

            lS = lZ - lI
            x, xerr = ms(sigma(p(lS, BMD)))
            sigmas[model][data] = x, xerr
            ax.errorbar(x, y, xerr=xerr, marker='o', color=color, label=label, alpha=0.9)

ax.set_yticks([i for i, data in enumerate(datasets) if len(data) > 0])
ax.set_yticklabels([' vs '.join(labels[x] for x in data)
                    for data in datasets if len(data) > 0])
ax.set_ylim(-1, len(datasets))

ax.set_xlabel('tension')
ax.set_xticks(numpy.arange(6))
ax.set_xticklabels([r'$%i\sigma$' % i for i in range(0, 6)])
ax.set_xlim(0, 5)

ax1 = ax.twiny()
ax1.set_xticks([sigma(10**(-i)) for i in range(8)])
ax1.set_xticklabels([r"%s\%%" % 10**(2-i) for i in range(8)])
ax1.set_xlabel('tension probability $p$')

ax1.set_xticks([sigma(x*10**(-i)) for i in range(8)
                for x in range(10, 1, -1)][9:], minor=True)
ax1.set_xlim(ax.get_xlim())
ax.xaxis.grid()
ax.yaxis.grid()

for i, data in enumerate(datasets):
    if len(data) == 0:
        ax.axhline(i, color='k', linewidth=1)

fig.tight_layout()
hands, labs = ax.get_legend_handles_labels()
dic = dict(zip(labs, hands))
fig.legend(list(dic.values()), list(dic), loc='upper left', ncol=2)

fig.savefig('tensions.pdf', bbox_inches='tight')


# Plotting evidences.pdf
fig, ax = plt.subplots()
fig.set_size_inches(4, 2.5)

datasets = [
            'lensing_BAO',
            'planck_BAO',
            'planck_lensing_BAO',
            'lensing_SH0ES',
            'planck_SH0ES',
            'planck_lensing_SH0ES',
            'planck_lensing',
            'BAO',
            'SH0ES',
            'lensing',
            'planck'
            ]
for y, data in enumerate(datasets):
    DlZ = stats['klcdm'][data].logZ - stats['lcdm'][data].logZ
    DD = stats['klcdm'][data].D - stats['lcdm'][data].D
    x, xerr = ms(DlZ)
    z, zerr = ms(DlZ + DD)
    ax.plot([x, z], [y, y], color='#ff7f0e', zorder=1000)
    ax.errorbar(x, y, xerr=xerr, marker='o', color='#1f77b4', zorder=1001)

ax.set_yticks(range(len(datasets)))
ax.set_yticklabels([labels[data] for data in datasets])
ax.set_ylim(-1, len(datasets))

ax.set_xlabel(r'$\Delta\log\mathcal{Z}$')

ax.axvline(0, color='k', linewidth=1)
ax.set_xticks(numpy.arange(-3, 7))
ax.xaxis.grid()
ax.yaxis.grid()

ax1 = ax.twiny()
xmin, xmax = ax.get_xlim()
ax1.set_xlim(numpy.exp(xmin), numpy.exp(xmax))
ax1.set_xscale('log')
ax1.set_xlabel('Betting odds (curved:flat)')

xticklabels = []
for x in ax1.get_xticks():
    if x < 1:
        text = r'$1:%i$' % int(1/x)
    elif x > 1:
        text = r'$%i:1$' % int(x)
    else:
        text = r'$1:1$'
    xticklabels.append(text)
ax1.set_xticklabels(xticklabels)


fig.tight_layout()
fig.savefig('evidences.pdf', bbox_inches='tight')


# Plotting omegak_H0.pdf
fig, axes = plt.subplots(1, 3, sharey=True)
fig.set_size_inches(7.5, 3)

klcdm = samples['klcdm']
prior = klcdm['lensing'].set_beta(0)
prior.label = r'prior'

for ax in axes:
    prior.plot(ax, 'omegak', 'H0', plot_type='scatter',
               ncompress=2000, zorder=-1000, color='lightgray')
    ax.set_xlabel(r'$\Omega_K$')
    ax.set_xticks([-0.1, -0.05, 0, 0.05])
    ax.set_xticklabels([r'$-10\%$', r'$-5\%$', r'$0\%$', r'$5\%$'])

axes[0].set_ylabel('$H_0$')

for ax in axes:
    klcdm['planck'].plot(ax, 'omegak', 'H0', ncompress=10000)

klcdm['lensing'].plot(axes[0], 'omegak', 'H0', ncompress=10000, zorder=-500)
axes[1].plot([], [])
axes[2].plot([], [])

klcdm['planck_lensing'].plot(axes[0], 'omegak', 'H0',
                             ncompress=10000, alpha=0.8)
axes[1].plot([], [])
axes[2].plot([], [])
axes[2].plot([], [])

axes[0].plot([], [])
klcdm['BAO'].plot(axes[1], 'omegak', 'H0', ncompress=10000)
axes[2].plot([], [])

axes[0].plot([], [])
klcdm['planck_BAO'].plot(axes[1], 'omegak', 'H0', ncompress=10000)
axes[2].plot([], [])

axes[0].plot([], [])
axes[1].plot([], [])
klcdm['SH0ES'].plot(axes[2], 'omegak', 'H0', ncompress=10000)

klcdm['planck_SH0ES'].plot(axes[2], 'omegak', 'H0', ncompress=10000)
axes[0].plot([], [])
axes[1].plot([], [])

for ax in axes:
    ax.legend(*ax.get_legend_handles_labels())
    ax.set_xlim(-0.11, 0.06)

for ax, dat in zip(axes, ['lensing', 'BAO', 'SH0ES']):
    ax.text(0.95, 0.05,
            r"$\sigma=%.2f \pm %.2f$" % sigmas['klcdm'][('planck', dat)],
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)

for ax in axes[1:]:
    ax.tick_params('y', left=False, labelleft=False)

fig.subplots_adjust(wspace=0, left=0.05, right=0.9, top=0.9, bottom=0.1)
fig.savefig('omegak_H0.pdf', bbox_inches='tight')


# Plotting constraints.pdf
params = ['omegak', 'omegabh2', 'omegach2', 'theta', 'tau', 'logA', 'ns']
fig, axes = make_2d_axes(params, tex=prior.tex)
fig.set_size_inches(7.05, 7.5)
for ax in axes.loc['ns', :]:
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
fig.tight_layout()
atypes = {'lower': 'kde', 'diagonal': 'kde', 'upper': 'kde'}
ltypes = {'lower': 'kde', 'diagonal': 'kde'}
utypes = {'upper': 'kde', 'diagonal': 'kde'}


prior.plot_2d(axes, types=atypes)
klcdm['BAO'].plot_2d(axes, alpha=0.9, types=utypes)
klcdm['lensing'].plot_2d(axes, alpha=0.9, types=ltypes)
klcdm['planck'].plot_2d(axes, alpha=0.9, types=atypes)
klcdm['planck_BAO'].plot_2d(axes, alpha=0.9, types=utypes)
klcdm['planck_lensing'].plot_2d(axes, alpha=0.9, types=ltypes)

lhandles, llabels = axes['omegak']['ns'].get_legend_handles_labels()
uhandles, ulabels = axes['ns']['omegak'].get_legend_handles_labels()
dic = dict(zip(llabels, lhandles))
dic.update(dict(zip(ulabels, uhandles)))

fig.legend(list(dic.values()), list(dic), loc='upper center', ncol=6,
           bbox_to_anchor=(0.5, 1.05))

fig.tight_layout()
fig.savefig('constraints.pdf', bbox_inches='tight')
