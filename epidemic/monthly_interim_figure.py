"""Static comparison including the temporal control that changes interpretation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    folder = Path('outputs/multi_month/interim_december')
    original = pd.read_csv(folder/'metrics.csv').query("scope == 'all'").set_index('predictor')
    controlled = pd.read_csv(folder/'calendar_controls/metrics.csv').query("scope == 'all'").set_index('predictor')
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x = np.arange(3)
    for offset, values, label, color in [
        (-.27, original, 'History + activity', '#8b969e'),
        (-.09, original, 'History + activity + sentiment', '#287d8e'),
        (.09, controlled, 'History + activity + calendar', '#b87832'),
        (.27, controlled, 'History + activity + calendar + sentiment', '#725996')]:
        name = 'combined_delta' if 'sentiment' in label else 'lag_activity_delta'
        errors = [100*values.loc[model+'_'+name, 'mae'] for model in ['ridge', 'forest', 'boosting']]
        ax.bar(x+offset, errors, width=.17, color=color, label=label)
    ax.set_xticks(x, ['Ridge', 'Random forest', 'Boosting'])
    ax.set_ylabel('Masking prediction MAE (percentage points; lower is better)')
    ax.set_ylim(0, 3.1)
    ax.legend(frameon=False, fontsize=9, loc='upper left', ncol=2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('December 2020 holdout: sentiment gains depend on the baseline\nTrain October–November; 386 counties; exploratory diagnostics', pad=14)
    fig.savefig(folder/'baseline_comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
