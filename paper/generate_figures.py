#!/usr/bin/env python3
"""
Visualization script for the HPO course report.
Run from the manu/ directory: python generate_figures.py
Outputs four PNG figures alongside this script.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# CONSISTENT COLOR SCHEME  (use these same colors everywhere)
# ============================================================
ARCH_COLORS = {
    'GAT':       '#4C72B0',   # blue
    'GCN':       '#DD8452',   # orange
    'GraphSAGE': '#8172B2',   # purple
    'GIN':       '#55A868',   # green
}

PARADIGM_COLORS = {
    'Exhaustive':    '#95A5A6',
    'Stochastic':    '#717D7E',
    'Model-based':   '#9B59B6',
    'Population':    '#1ABC9C',
    'Evolution':     '#1E8449',
    'Local Search':  '#E74C3C',
    'Bandit':        '#F39C12',
}

ALG_PARADIGM = {
    'Grid Search':         'Exhaustive',
    'Random Search':       'Stochastic',
    'Bayesian':            'Model-based',
    'ABC':                 'Population',
    'Genetic':             'Population',
    'PSO':                 'Population',
    'CMA-ES':              'Evolution',
    'Hill Climbing':       'Local Search',
    'Simulated Annealing': 'Local Search',
    'Tabu Search':         'Local Search',
    'Hyperband':           'Bandit',
}

ALG_MARKERS = {
    'Grid Search':         's',
    'Random Search':       'D',
    'Bayesian':            'P',
    'ABC':                 'o',
    'Genetic':             '^',
    'PSO':                 'v',
    'CMA-ES':              '*',
    'Hill Climbing':       'X',
    'Simulated Annealing': 'h',
    'Tabu Search':         'H',
    'Hyperband':           'p',
}

# ============================================================
# DATA
# ============================================================
# Order matches ALGORITHMS list below (by paradigm)
ALGORITHMS = [
    'Grid Search', 'Random Search',
    'Bayesian',
    'ABC', 'Genetic', 'PSO', 'CMA-ES',
    'Hill Climbing', 'Simulated Annealing', 'Tabu Search',
    'Hyperband',
]

ARCHITECTURES = ['GAT', 'GCN', 'GraphSAGE', 'GIN']

# Proxy validation F1 (10% data, 10 epochs)
PROXY_F1 = {
    'GAT':       [0.8814, 0.8699, 0.8537, 0.8440, 0.8245, 0.8623, 0.8693, 0.8097, 0.8675, 0.8769, 0.7775],
    'GCN':       [0.8383, 0.8440, 0.7211, 0.7576, 0.7569, 0.7274, 0.7477, 0.7771, 0.8403, 0.8499, 0.7168],
    'GraphSAGE': [0.8641, 0.8715, 0.8267, 0.8383, 0.8486, 0.8611, 0.8710, 0.8201, 0.8918, 0.8955, 0.7919],
    'GIN':       [0.9244, 0.9162, 0.8667, 0.8872, 0.8723, 0.8874, 0.8798, 0.8874, 0.9159, 0.9257, 0.8752],
}

# Total proxy-phase evaluations per algorithm
EVALUATIONS = {
    'Grid Search':         100,
    'Random Search':       100,
    'Bayesian':            100,
    'Genetic':             650,
    'PSO':                 620,
    'ABC':                 1270,
    'CMA-ES':              500,
    'Hill Climbing':       430,
    'Simulated Annealing': 153,
    'Tabu Search':         1350,
    'Hyperband':           100,
}

# Full training test metrics -- all 11 algorithms x 4 architectures
ALL_FULL = {
    'GAT': {
        'Grid Search':         {'Acc': 0.7532, 'Prec': 0.7500, 'Rec': 0.8924, 'Spec': 0.5362, 'F1': 0.8150, 'AUC': 0.7893, 'MCC': 0.4683},
        'Random Search':       {'Acc': 0.7802, 'Prec': 0.7701, 'Rec': 0.9114, 'Spec': 0.5757, 'F1': 0.8348, 'AUC': 0.8162, 'MCC': 0.5299},
        'Bayesian':            {'Acc': 0.8072, 'Prec': 0.7862, 'Rec': 0.9388, 'Spec': 0.6020, 'F1': 0.8558, 'AUC': 0.8627, 'MCC': 0.5926},
        'ABC':                 {'Acc': 0.7789, 'Prec': 0.7603, 'Rec': 0.9304, 'Spec': 0.5428, 'F1': 0.8368, 'AUC': 0.8405, 'MCC': 0.5300},
        'Genetic':             {'Acc': 0.7661, 'Prec': 0.7552, 'Rec': 0.9114, 'Spec': 0.5395, 'F1': 0.8260, 'AUC': 0.8114, 'MCC': 0.4986},
        'PSO':                 {'Acc': 0.8175, 'Prec': 0.7986, 'Rec': 0.9367, 'Spec': 0.6316, 'F1': 0.8621, 'AUC': 0.8800, 'MCC': 0.6140},
        'CMA-ES':              {'Acc': 0.8136, 'Prec': 0.8007, 'Rec': 0.9241, 'Spec': 0.6414, 'F1': 0.8580, 'AUC': 0.8735, 'MCC': 0.6039},
        'Hill Climbing':       {'Acc': 0.7622, 'Prec': 0.7576, 'Rec': 0.8966, 'Spec': 0.5526, 'F1': 0.8213, 'AUC': 0.8179, 'MCC': 0.4888},
        'Simulated Annealing': {'Acc': 0.7391, 'Prec': 0.7673, 'Rec': 0.8207, 'Spec': 0.6118, 'F1': 0.7931, 'AUC': 0.7601, 'MCC': 0.4429},
        'Tabu Search':         {'Acc': 0.7802, 'Prec': 0.7750, 'Rec': 0.9008, 'Spec': 0.5921, 'F1': 0.8332, 'AUC': 0.8379, 'MCC': 0.5291},
        'Hyperband':           {'Acc': 0.7853, 'Prec': 0.7717, 'Rec': 0.9198, 'Spec': 0.5757, 'F1': 0.8393, 'AUC': 0.8403, 'MCC': 0.5422},
    },
    'GCN': {
        'Grid Search':         {'Acc': 0.7429, 'Prec': 0.7299, 'Rec': 0.9177, 'Spec': 0.4704, 'F1': 0.8131, 'AUC': 0.7977, 'MCC': 0.4473},
        'Random Search':       {'Acc': 0.7326, 'Prec': 0.7145, 'Rec': 0.9346, 'Spec': 0.4178, 'F1': 0.8099, 'AUC': 0.7626, 'MCC': 0.4274},
        'Bayesian':            {'Acc': 0.7134, 'Prec': 0.7495, 'Rec': 0.7954, 'Spec': 0.5855, 'F1': 0.7718, 'AUC': 0.7300, 'MCC': 0.3887},
        'ABC':                 {'Acc': 0.7018, 'Prec': 0.7490, 'Rec': 0.7679, 'Spec': 0.5987, 'F1': 0.7583, 'AUC': 0.7246, 'MCC': 0.3694},
        'Genetic':             {'Acc': 0.7211, 'Prec': 0.7495, 'Rec': 0.8143, 'Spec': 0.5757, 'F1': 0.7806, 'AUC': 0.7359, 'MCC': 0.4023},
        'PSO':                 {'Acc': 0.6607, 'Prec': 0.6640, 'Rec': 0.8966, 'Spec': 0.2928, 'F1': 0.7630, 'AUC': 0.6659, 'MCC': 0.2419},
        'CMA-ES':              {'Acc': 0.7237, 'Prec': 0.7376, 'Rec': 0.8481, 'Spec': 0.5296, 'F1': 0.7890, 'AUC': 0.7326, 'MCC': 0.4024},
        'Hill Climbing':       {'Acc': 0.7134, 'Prec': 0.7131, 'Rec': 0.8861, 'Spec': 0.4441, 'F1': 0.7902, 'AUC': 0.7343, 'MCC': 0.3756},
        'Simulated Annealing': {'Acc': 0.7622, 'Prec': 0.7445, 'Rec': 0.9283, 'Spec': 0.5033, 'F1': 0.8263, 'AUC': 0.7883, 'MCC': 0.4928},
        'Tabu Search':         {'Acc': 0.7506, 'Prec': 0.7439, 'Rec': 0.9008, 'Spec': 0.5164, 'F1': 0.8149, 'AUC': 0.8183, 'MCC': 0.4629},
        'Hyperband':           {'Acc': 0.6581, 'Prec': 0.6576, 'Rec': 0.9156, 'Spec': 0.2566, 'F1': 0.7654, 'AUC': 0.6587, 'MCC': 0.2342},
    },
    'GraphSAGE': {
        'Grid Search':         {'Acc': 0.7892, 'Prec': 0.7768, 'Rec': 0.9177, 'Spec': 0.5888, 'F1': 0.8414, 'AUC': 0.8478, 'MCC': 0.5503},
        'Random Search':       {'Acc': 0.7365, 'Prec': 0.7307, 'Rec': 0.8987, 'Spec': 0.4836, 'F1': 0.8061, 'AUC': 0.7708, 'MCC': 0.4304},
        'Bayesian':            {'Acc': 0.7725, 'Prec': 0.7565, 'Rec': 0.9241, 'Spec': 0.5362, 'F1': 0.8319, 'AUC': 0.8596, 'MCC': 0.5147},
        'ABC':                 {'Acc': 0.8008, 'Prec': 0.7823, 'Rec': 0.9325, 'Spec': 0.5954, 'F1': 0.8508, 'AUC': 0.8890, 'MCC': 0.5776},
        'Genetic':             {'Acc': 0.7789, 'Prec': 0.7687, 'Rec': 0.9113, 'Spec': 0.5724, 'F1': 0.8340, 'AUC': 0.8568, 'MCC': 0.5271},
        'PSO':                 {'Acc': 0.7931, 'Prec': 0.7893, 'Rec': 0.9008, 'Spec': 0.6250, 'F1': 0.8414, 'AUC': 0.8654, 'MCC': 0.5575},
        'CMA-ES':              {'Acc': 0.7995, 'Prec': 0.8058, 'Rec': 0.8840, 'Spec': 0.6678, 'F1': 0.8431, 'AUC': 0.8850, 'MCC': 0.5718},
        'Hill Climbing':       {'Acc': 0.7969, 'Prec': 0.7894, 'Rec': 0.9093, 'Spec': 0.6217, 'F1': 0.8451, 'AUC': 0.8665, 'MCC': 0.5663},
        'Simulated Annealing': {'Acc': 0.7905, 'Prec': 0.7714, 'Rec': 0.9325, 'Spec': 0.5691, 'F1': 0.8443, 'AUC': 0.8831, 'MCC': 0.5555},
        'Tabu Search':         {'Acc': 0.8059, 'Prec': 0.8030, 'Rec': 0.9030, 'Spec': 0.6546, 'F1': 0.8500, 'AUC': 0.8824, 'MCC': 0.5857},
        'Hyperband':           {'Acc': 0.7674, 'Prec': 0.7575, 'Rec': 0.9093, 'Spec': 0.5461, 'F1': 0.8265, 'AUC': 0.8496, 'MCC': 0.5012},
    },
    'GIN': {
        'Grid Search':         {'Acc': 0.8368, 'Prec': 0.8548, 'Rec': 0.8819, 'Spec': 0.7664, 'F1': 0.8681, 'AUC': 0.9105, 'MCC': 0.6546},
        'Random Search':       {'Acc': 0.8355, 'Prec': 0.8264, 'Rec': 0.9241, 'Spec': 0.6974, 'F1': 0.8725, 'AUC': 0.8988, 'MCC': 0.6506},
        'Bayesian':            {'Acc': 0.8509, 'Prec': 0.8580, 'Rec': 0.9051, 'Spec': 0.7664, 'F1': 0.8809, 'AUC': 0.9063, 'MCC': 0.6837},
        'ABC':                 {'Acc': 0.8508, 'Prec': 0.8432, 'Rec': 0.8734, 'Spec': 0.7467, 'F1': 0.8580, 'AUC': 0.9068, 'MCC': 0.6271},
        'Genetic':             {'Acc': 0.7802, 'Prec': 0.7546, 'Rec': 0.9473, 'Spec': 0.5197, 'F1': 0.8400, 'AUC': 0.8737, 'MCC': 0.5372},
        'PSO':                 {'Acc': 0.8496, 'Prec': 0.8549, 'Rec': 0.9072, 'Spec': 0.7599, 'F1': 0.8802, 'AUC': 0.9122, 'MCC': 0.6808},
        'CMA-ES':              {'Acc': 0.8586, 'Prec': 0.8583, 'Rec': 0.9198, 'Spec': 0.7632, 'F1': 0.8880, 'AUC': 0.9307, 'MCC': 0.7000},
        'Hill Climbing':       {'Acc': 0.8548, 'Prec': 0.8915, 'Rec': 0.8671, 'Spec': 0.8355, 'F1': 0.8791, 'AUC': 0.9187, 'MCC': 0.6977},
        'Simulated Annealing': {'Acc': 0.8329, 'Prec': 0.8258, 'Rec': 0.9198, 'Spec': 0.6974, 'F1': 0.8703, 'AUC': 0.9116, 'MCC': 0.6449},
        'Tabu Search':         {'Acc': 0.8509, 'Prec': 0.8580, 'Rec': 0.9051, 'Spec': 0.7664, 'F1': 0.8809, 'AUC': 0.9166, 'MCC': 0.6837},
        'Hyperband':           {'Acc': 0.8419, 'Prec': 0.8619, 'Rec': 0.8819, 'Spec': 0.7796, 'F1': 0.8717, 'AUC': 0.9118, 'MCC': 0.6661},
    },
}

ALL_ALGS = list(ALGORITHMS)  # all 11

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
})


# ============================================================
# FIGURE 1: Proxy F1 Heatmap (11 algorithms x 4 architectures)
# ============================================================
def fig1_proxy_heatmap():
    fig, ax = plt.subplots(figsize=(7, 7.5))

    matrix = np.array([[PROXY_F1[arch][i] for arch in ARCHITECTURES]
                        for i in range(len(ALGORITHMS))])

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.70, vmax=0.94, aspect='auto')

    ax.set_xticks(range(len(ARCHITECTURES)))
    ax.set_xticklabels(ARCHITECTURES, fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(ALGORITHMS)))
    ax.set_yticklabels(ALGORITHMS, fontsize=10)

    for i in range(len(ALGORITHMS)):
        for j in range(len(ARCHITECTURES)):
            val = matrix[i, j]
            txt_color = 'white' if val < 0.76 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8.5, color=txt_color, fontweight='bold')

    for sep in [1.5, 2.5, 6.5, 10.5]:
        ax.axhline(sep, color='white', linewidth=2.5)

    for i, alg in enumerate(ALGORITHMS):
        ax.get_yticklabels()[i].set_color(PARADIGM_COLORS[ALG_PARADIGM[alg]])

    for j, arch in enumerate(ARCHITECTURES):
        ax.get_xticklabels()[j].set_color(ARCH_COLORS[arch])

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Proxy Val F1', fontsize=9)

    paradigm_spans = [
        ('Exhaustive /\nStochastic', 0, 1),
        ('Model-Based',             2, 2),
        ('Population',              3, 6),
        ('Local Search',            7, 9),
        ('Bandit',                  10, 10),
    ]
    for label, start, end in paradigm_spans:
        mid = (start + end) / 2
        ax.annotate(label, xy=(1.18, 1 - (mid + 0.5) / len(ALGORITHMS)),
                    xycoords='axes fraction', fontsize=7.5, ha='left', va='center',
                    color='gray', rotation=0)

    ax.set_title('Proxy Validation F1 -- All Algorithms and Architectures\n'
                 '(10% of B3DB, 10 training epochs)', fontsize=11, pad=8)
    ax.tick_params(axis='x', which='both', bottom=False, top=True, labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position('top')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig1_proxy_heatmap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ============================================================
# FIGURE 2: Budget vs Quality scatter (efficiency frontier)
# ============================================================
def fig2_budget_quality():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    gin_f1 = {alg: PROXY_F1['GIN'][i] for i, alg in enumerate(ALGORITHMS)}
    evals  = {alg: EVALUATIONS[alg] for alg in ALGORITHMS}

    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

    for alg in ALGORITHMS:
        x = evals[alg]
        y = gin_f1[alg]
        color  = PARADIGM_COLORS[ALG_PARADIGM[alg]]
        marker = ALG_MARKERS[alg]
        ax.scatter(x, y, color=color, marker=marker, s=120, zorder=3,
                   edgecolors='white', linewidths=0.6)

    pts = sorted([(evals[a], gin_f1[a], a) for a in ALGORITHMS], key=lambda t: t[0])
    pareto_x, pareto_y = [], []
    best_y = -1
    for x, y, _ in pts:
        if y > best_y:
            pareto_x.append(x)
            pareto_y.append(y)
            best_y = y
    ax.plot(pareto_x, pareto_y, color='black', lw=1.2, ls='--', alpha=0.5,
            zorder=2, label='Pareto frontier')

    offsets = {
        'Grid Search':         (-10,  6),
        'Random Search':       (-10, -12),
        'Bayesian':            (-10,  6),
        'Hyperband':           (6,   -12),
        'Simulated Annealing': (6,    4),
        'Hill Climbing':       (6,    4),
        'CMA-ES':              (6,    4),
        'PSO':                 (-10, -12),
        'Genetic':             (6,   -12),
        'ABC':                 (6,    4),
        'Tabu Search':         (6,    4),
    }
    for alg in ALGORITHMS:
        x = evals[alg]
        y = gin_f1[alg]
        dx, dy = offsets.get(alg, (6, 4))
        ax.annotate(alg, xy=(x, y), xytext=(dx, dy),
                    textcoords='offset points', fontsize=8,
                    color=PARADIGM_COLORS[ALG_PARADIGM[alg]],
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.6),
                    ha='left' if dx > 0 else 'right')

    handles = [mpatches.Patch(color=PARADIGM_COLORS[p], label=p)
               for p in ['Exhaustive', 'Stochastic', 'Model-based',
                         'Population', 'Evolution', 'Local Search', 'Bandit']]
    ax.legend(handles=handles, loc='lower right', fontsize=8,
              framealpha=0.9, title='Paradigm', title_fontsize=8)

    ax.set_xlabel('Total Proxy Evaluations', fontsize=10)
    ax.set_ylabel('Best Proxy Val F1 on GIN', fontsize=10)
    ax.set_title('HPO Efficiency: Evaluation Budget vs. Proxy F1 (GIN)\n'
                 'Points on the dashed line are Pareto-optimal', fontsize=11)
    ax.set_xlim(left=60)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig2_budget_quality.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ============================================================
# FIGURE 3: Architecture sensitivity -- box plots
# ============================================================
def fig3_arch_sensitivity():
    fig, ax = plt.subplots(figsize=(7, 5))

    data_by_arch = {arch: PROXY_F1[arch] for arch in ARCHITECTURES}

    positions = range(len(ARCHITECTURES))
    bp = ax.boxplot(
        [data_by_arch[arch] for arch in ARCHITECTURES],
        positions=list(positions),
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=5, linestyle='none'),
        zorder=3,
    )

    for patch, arch in zip(bp['boxes'], ARCHITECTURES):
        patch.set_facecolor(ARCH_COLORS[arch])
        patch.set_alpha(0.75)
    for flier, arch in zip(bp['fliers'], ARCHITECTURES):
        flier.set_markerfacecolor(ARCH_COLORS[arch])
        flier.set_markeredgecolor(ARCH_COLORS[arch])

    rng = np.random.default_rng(42)
    for j, arch in enumerate(ARCHITECTURES):
        vals = data_by_arch[arch]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(j + jitter, vals, color=ARCH_COLORS[arch],
                   s=30, alpha=0.8, zorder=4, edgecolors='white', linewidths=0.4)

    for j, arch in enumerate(ARCHITECTURES):
        vals = data_by_arch[arch]
        rng_val = max(vals) - min(vals)
        ax.text(j, min(vals) - 0.008, f'range\n{rng_val:.3f}',
                ha='center', va='top', fontsize=7.5, color=ARCH_COLORS[arch])

    ax.set_xticks(list(positions))
    ax.set_xticklabels(ARCHITECTURES, fontsize=11, fontweight='bold')
    for tick, arch in zip(ax.get_xticklabels(), ARCHITECTURES):
        tick.set_color(ARCH_COLORS[arch])

    ax.set_ylabel('Proxy Val F1 (across all 11 algorithms)', fontsize=10)
    ax.set_title('Architecture Sensitivity to HPO Algorithm Choice\n'
                 'Each point is one of the eleven optimizers', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(0.69, 0.95)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig3_arch_sensitivity.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ============================================================
# FIGURE 4: AUC vs MCC scatter -- all 11 algorithms x 4 architectures
# ============================================================
def fig4_auc_mcc_scatter():
    fig, ax = plt.subplots(figsize=(9, 6.5))

    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

    for arch in ARCHITECTURES:
        for alg in ALL_ALGS:
            d = ALL_FULL[arch][alg]
            auc = d['AUC']
            mcc = d['MCC']
            marker = ALG_MARKERS[alg]
            ax.scatter(auc, mcc, color=ARCH_COLORS[arch], marker=marker,
                       s=100, zorder=3, edgecolors='white', linewidths=0.7,
                       alpha=0.9)

    from matplotlib.patches import Ellipse
    for arch in ARCHITECTURES:
        aucs = [ALL_FULL[arch][a]['AUC'] for a in ALL_ALGS]
        mccs = [ALL_FULL[arch][a]['MCC'] for a in ALL_ALGS]
        cx, cy = np.mean(aucs), np.mean(mccs)
        w  = (max(aucs) - min(aucs)) * 1.3 + 0.015
        h  = (max(mccs) - min(mccs)) * 1.3 + 0.015
        el = Ellipse((cx, cy), width=w, height=h,
                     color=ARCH_COLORS[arch], alpha=0.10, zorder=1)
        ax.add_patch(el)
        ax.text(cx, cy + h / 2 + 0.012, arch, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=ARCH_COLORS[arch])

    arch_handles = [mpatches.Patch(color=ARCH_COLORS[a], label=a) for a in ARCHITECTURES]
    alg_handles = [Line2D([0], [0], marker=ALG_MARKERS[a], color='black', linestyle='none',
                          markersize=7, label=a) for a in ALL_ALGS]

    leg1 = ax.legend(handles=arch_handles, loc='lower right', fontsize=8,
                     title='Architecture', title_fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=alg_handles, loc='lower left', fontsize=7.5,
              title='Optimizer', title_fontsize=8, framealpha=0.9, ncol=2)

    ax.set_xlabel('ROC-AUC (full training, B3DB test set)', fontsize=10)
    ax.set_ylabel('MCC (full training, B3DB test set)', fontsize=10)
    ax.set_title('AUC vs. MCC for All 44 Architecture-Optimizer Combinations\n'
                 '(11 algorithms x 4 architectures, 100% data, 100 epochs)', fontsize=11)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig4_auc_mcc_scatter.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ============================================================
# FIGURE 5: Best per-architecture bar chart (F1, AUC, MCC)
# ============================================================
def _lighten(hex_color, amount):
    """Blend hex_color toward white by `amount` (0=original, 1=white)."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hex_color)
    return (r + (1 - r) * amount,
            g + (1 - g) * amount,
            b + (1 - b) * amount)


def fig_best_results():
    BEST = {
        'GIN':       ('CMA-ES',              0.8880, 0.9307, 0.7000),
        'GraphSAGE': ('ABC',                 0.8508, 0.8890, 0.5776),
        'GAT':       ('PSO',                 0.8621, 0.8800, 0.6140),
        'GCN':       ('Simulated Annealing', 0.8263, 0.7883, 0.4928),
    }
    order = ['GIN', 'GraphSAGE', 'GAT', 'GCN']

    # AUC: full color, solid
    # F1:  lightened color, diagonal hatch in same lightened color
    # MCC: more lightened, cross hatch in same color
    METRIC_IDX  = {'AUC': 2, 'F1': 1, 'MCC': 3}
    hatches     = {'AUC': '',     'F1': '///',  'MCC': 'xxx'}
    lighten_amt = {'AUC': 0.0,   'F1': 0.35,   'MCC': 0.58}

    x      = np.arange(len(order))
    width  = 0.26
    offsets = [-width, 0, width]
    metrics = ['AUC', 'F1', 'MCC']

    fig, ax = plt.subplots(figsize=(8, 5.2))

    for i, metric in enumerate(metrics):
        vals = [BEST[arch][METRIC_IDX[metric]] for arch in order]
        for j, (arch, val) in enumerate(zip(order, vals)):
            base = ARCH_COLORS[arch]
            fill = _lighten(base, lighten_amt[metric])
            # fill is lightened; hatch lines use the base color so they're
            # visible against the lighter background but still in-family
            hatch_color = _lighten(base, lighten_amt[metric] * 0.25)
            ax.bar(
                x[j] + offsets[i], val, width,
                facecolor=fill,
                hatch=hatches[metric],
                edgecolor=hatch_color,
                linewidth=0.6,
                zorder=3,
            )
            # value label
            ax.text(
                x[j] + offsets[i],
                val + 0.006,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontsize=7.0, fontweight='bold',
                color=_lighten(base, 0.1),
            )

    labels = [f'{a}\n({BEST[a][0]})' for a in order]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    for tick, arch in zip(ax.get_xticklabels(), order):
        tick.set_color(ARCH_COLORS[arch])

    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title('Best Full-Training Test Results per Architecture\n'
                 'Ordered by AUC; best optimizer shown in parentheses', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    # Legend using one neutral color so pattern is the focus
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#888888', edgecolor='#888888', linewidth=0,
              label='AUC  (solid)'),
        Patch(facecolor='#aaaaaa', edgecolor='#aaaaaa', linewidth=0,
              hatch='///', label='F1    (diagonal)'),
        Patch(facecolor='#cccccc', edgecolor='#cccccc', linewidth=0,
              hatch='xxx', label='MCC (cross)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_best_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


if __name__ == '__main__':
    print('Generating figures...')
    fig3_arch_sensitivity()
    fig_best_results()
    print('Done. Two PNG files saved to:', OUTPUT_DIR)
