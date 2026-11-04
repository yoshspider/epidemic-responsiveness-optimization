"""Presentation-ready synthetic mechanism figures, no report or slides."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out = Path('outputs/ideal_information_benchmark_84d')
    data = pd.read_csv(out / 'summary.csv')
    meta = json.loads((out / 'manifest.json').read_text())
    central = data[(data.average_dose == 1) & (data.mobility == 0)].set_index('policy')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    values = [meta['no_intervention_infections']] + [central.loc[name, 'infections']
                                                     for name in ['prior_optimal', 'noisy_measurement_sd0.15', 'perfect_information']]
    axes[0].bar(
        np.arange(4),
        np.array(values) / 1e6,
        color=[
            '#b9c0c9',
            '#7d8da6',
            '#e4a555',
            '#147d80'])
    axes[0].set_xticks(np.arange(4), ['No\nintervention', 'Distribution\nonly',
                       'Noisy county\nmeasurement', 'Exact county\nresponsiveness'])
    axes[0].set_ylabel('New infections over 12 weeks (millions)')
    axes[0].set_title('Intervention works; information adds substantial value')
    for index, value in enumerate(values):
        axes[0].text(index, value / 1e6 + .05, f'{value / 1e6:.3f}m', ha='center', fontsize=10)
    axes[0].set_ylim(0, max(values) / 1e6 * 1.15)
    for mobility, label, color in [(0, 'Independent counties', '#147d80'), (.02,
                                                                            '2% mobility mixing', '#d18b33'), (.1, '10% mobility mixing', '#7354a0')]:
        part = data[(data.mobility == mobility) & (
            data.policy == 'perfect_information')].sort_values('average_dose')
        axes[1].plot(part.average_dose, part.improvement_percent, 'o-', label=label, color=color)
    axes[1].set_xlabel('Average sustained intervention dose (normalized units)')
    axes[1].set_ylabel('Extra reduction from exact county information (%)')
    axes[1].set_title('Benefit across budgets and mobility settings')
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.07, .015, 'Fully synthetic: 32 equal-risk counties, Uniform(0,1) responsiveness, 24 independent worlds.\nSame budgets; optimal static grid plans for independent counties. Noisy measurement SD=0.15; not empirical survey accuracy.', fontsize=9)
    fig.tight_layout(rect=[0, .11, 1, 1])
    fig.savefig(out / 'ideal_information_results.png', dpi=180)
    # Quality curve at the central budget, evaluated with and without mobility.
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    names = [
        'prior_optimal',
        'noisy_measurement_sd0.3',
        'noisy_measurement_sd0.15',
        'noisy_measurement_sd0.05',
        'perfect_information']
    for mobility, color, label in [(0, '#147d80', 'Independent counties'),
                                   (.1, '#7354a0', '10% mobility mixing')]:
        table = data[(data.average_dose == 1) & (data.mobility == mobility)].set_index('policy')
        ax.plot(np.arange(5), [table.loc[name, 'improvement_percent']
                for name in names], 'o-', label=label, color=color)
    ax.set_xticks(np.arange(5), ['Distribution\nonly', 'Noise SD\n0.30',
                  'Noise SD\n0.15', 'Noise SD\n0.05', 'Perfect\ninformation'])
    ax.set_ylabel('Reduction versus distribution-only allocation (%)')
    ax.set_title('More accurate responsiveness information improves allocation')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.12, .015, 'Controlled synthetic measurements; Bayesian planning given a known noise model.\n12-week epidemic; central budget; 24 random responsiveness maps.', fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, 1])
    fig.savefig(out / 'information_quality_curve.png', dpi=180)


if __name__ == '__main__':
    main()
