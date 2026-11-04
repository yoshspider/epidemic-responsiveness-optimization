"""Publication-style figures for synthetic prediction and policy results."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    folder = Path('outputs/learnable_behavior')
    comparisons = pd.read_csv(folder/'comparisons.csv')
    predictions = pd.read_csv(folder/'predictions.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, scenario in zip(axes, ['null', 'moderate', 'strong']):
        group = comparisons[comparisons.scenario == scenario]
        for policy, offset, color in [('predicted', -.012, '#147d92'), ('oracle', .012, '#d17a28')]:
            data = group[group.policy == policy]
            scale = data.relative_percent / data.additional_infections_prevented.replace(0, 1)
            ax.errorbar(data.budget_fraction+offset, data.relative_percent,
                        yerr=[(data.additional_infections_prevented-data.ci_low)*scale,
                              (data.ci_high-data.additional_infections_prevented)*scale],
                        fmt='o', capsize=4, label=policy, color=color)
        ax.axhline(0, color='gray', lw=.8)
        ax.set_title(scenario.capitalize()+' responsiveness')
        ax.set_xticks([.05, .2], ['5%', '20%'])
        ax.set_xlabel('Weekly budget / full-coverage effort cost')
        ax.set_ylim(-.5, 8)
    axes[0].set_ylabel('Additional infections prevented (%)')
    axes[2].legend(frameon=False)
    fig.suptitle('Fully synthetic: learned behavior captures most oracle benefit\n84 days; 16 fresh test worlds; paired 95% t intervals')
    fig.savefig(folder/'policy_comparison.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, scenario in zip(axes, ['moderate', 'strong']):
        group = predictions[predictions.scenario == scenario]
        ax.scatter(group.truth, group.prediction, s=10, alpha=.45, color='#147d92')
        ax.plot([0, 1], [0, 1], color='gray', lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='True latent responsiveness',
               ylabel='Held-out predicted responsiveness', title=scenario.capitalize())
    fig.suptitle('Synthetic prediction: noisy training labels, separate test worlds')
    fig.savefig(folder/'prediction_comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
