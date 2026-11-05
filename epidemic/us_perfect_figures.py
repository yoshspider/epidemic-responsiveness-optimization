"""Scientific figures for the held-out pure-information U.S. benchmark."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from .us_proxy_figures import geometry, draw, mapping
from .responsiveness_mechanisms import force


def trajectory(n, s0, i0, p, plan, matrices):
    s = s0.copy()
    i = i0.copy()
    removed = np.zeros(len(n))
    cumulative = np.zeros(len(n))
    weeks = []
    for day in range(84):
        week = day // 7
        dose = plan[week] if plan.ndim == 2 else plan
        h = force(i, n, p, dose, matrices[week], .36, .15, 'recipient')
        new = s * (-np.expm1(-h))
        recovered = i * (-np.expm1(-1 / 7))
        s -= new
        i += new - recovered
        removed += recovered
        cumulative += new
        if day % 7 == 6:
            weeks.append(cumulative.copy())
    np.testing.assert_allclose(s + i + removed, n)
    return np.asarray(weeks)


def main():
    root = Path('outputs/us_perfect_information')
    out = root / 'confirmation'
    summary = pd.read_csv(out / 'summary.csv').set_index('policy')
    trials = pd.read_csv(out / 'trials.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = ['No\nintervention', 'Uninformed\nallocation', 'Perfect county\nresponsiveness']
    policies = ['none', 'uninformed', 'perfect']
    prevention = [summary.loc[p, 'prevention_percent'] for p in policies]
    bars = axes[0].bar(range(3), prevention, color=['#aab4c0', '#7a8fae', '#147d80'])
    for index, value in enumerate(prevention):
        axes[0].text(index, value + 1.2, f'{value:.1f}%', ha='center', fontsize=12)
    axes[0].set_xticks(range(3), labels)
    axes[0].set_ylim(0, 65)
    axes[0].set_ylabel('Infections prevented versus no intervention (%)')
    axes[0].set_title('Both intervention and responsiveness matter')
    wide = trials.pivot(index='world', columns='policy', values='infections')
    relative = 100 * (wide.uninformed - wide.perfect) / wide.uninformed
    axes[1].scatter(np.arange(1, 17), relative, color='#147d80', s=38)
    axes[1].axhline(10, color='#8a8f98', linestyle='--', label='10% reference')
    axes[1].axhline(summary.loc['perfect', 'relative_reduction_percent'],
                    color='#147d80', linewidth=1, label='Aggregate17.6%')
    axes[1].set_xlabel('Fresh responsiveness map')
    axes[1].set_ylabel('Reduction versus uninformed allocation (%)')
    axes[1].set_ylim(0, 23)
    axes[1].set_xticks([1, 4, 8, 12, 16])
    axes[1].set_title('Benefit holds in every confirmation world')
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Perfect-information validation on the U.S. county network', fontsize=15)
    fig.text(.07, .015, 'Actual county populations and weekly mobility; synthetic Uniform(0,1) responsiveness and initial epidemic.\n84days; matched weekly budgets; frozen uninformed comparator;16 independent held-out response maps. No empirical proxy tested.', fontsize=9)
    fig.tight_layout(rect=[0, .1, 1, .94])
    fig.savefig(out / 'perfect_information_validation.png', dpi=180)
    plt.close(fig)
    initial = np.load('outputs/us_proxy_study/geographic/initial.npz')
    n = initial['n']
    counties = pd.read_csv('outputs/us_proxy_study/geographic/counties.csv', dtype={'fips': str})
    matrices = [
        sparse.load_npz(
            f'outputs/us_proxy_study/geographic/mobility_{week}.npz') for week in range(12)]
    case = root / 'recipient_0.36_0.8_0.15_1011000_stratified_weekly'
    baseline = np.load(case / 'selected_baseline.npz')['dose']
    oracle = np.load(case / 'oracle_0.npz')
    p = oracle['p']
    a = trajectory(n, initial['s'], initial['i'], p, baseline, matrices)
    b = trajectory(n, initial['s'], initial['i'], p, oracle['dose'], matrices)
    np.savez_compressed(out / 'example_county_trajectories.npz', p=p, uninformed=a, perfect=b)
    geo = geometry()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    first = draw(axes[0], geo, mapping(counties, p), Normalize(0, 1), 'viridis')
    axes[0].set_title('Synthetic true responsiveness, 0–100%', fontsize=12)
    fig.colorbar(first, ax=axes[0], fraction=.025, pad=.01, label='Responsiveness p')
    delta = (a[-1] - b[-1]) / n * 100000
    limit = max(1, float(np.quantile(np.abs(delta), .99)))
    second = draw(
        axes[1], geo, mapping(
            counties, delta), SymLogNorm(
            linthresh=100, vmin=-limit, vmax=limit), 'RdBu')
    axes[1].set_title('Additional infections avoided with perfect information', fontsize=12)
    fig.colorbar(second, ax=axes[1], fraction=.025, pad=.01,
                 label='Per100,000; blue=fewer, red=more')
    fig.suptitle('Where responsiveness information changes the epidemic', fontsize=15)
    fig.text(.07, .025, 'First held-out response map, not selected for performance. Synthetic responsiveness—not an observed survey map.\nBoth policies use the same weekly budget; county gains and losses are displayed; difference color scale capped at99th percentile.', fontsize=9)
    fig.subplots_adjust(top=.84, bottom=.13, left=.02, right=.94, wspace=.19)
    fig.savefig(out / 'perfect_information_county_map.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Saved held-out validation and county benefit figures', flush=True)


if __name__ == '__main__':
    main()
