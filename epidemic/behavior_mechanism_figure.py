"""Matched intervention strength makes mechanism comparisons interpretable."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('outputs/behavior_mechanism_compare');data=pd.read_csv(out/'summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for mechanism,label,color in [('direct','Direct suppression','#147d80'),('adoption','Persistent adoption','#b56b38')]:
        subset=data[(data.mechanism==mechanism)&(data.policy=='oracle')].sort_values('target')
        axes[0].plot(100*subset.target,subset.gain_percent,'o-',label=label,color=color)
        part=data[(data.mechanism==mechanism)&(data.target==.5)].set_index('policy')
        for offset,policy in [(-.16,'aggregate'),(.16,'oracle')]:
            index=0 if mechanism=='direct' else 1
            axes[1].bar(index+offset,part.loc[policy,'prevention_percent'],width=.28,color='#9ba7b4' if policy=='aggregate' else color,
                label=('Average information' if policy=='aggregate' else 'Full county information') if mechanism=='direct' else None)
    axes[0].set_xlabel('Reference intervention effect matched across models (%)')
    axes[0].set_ylabel('Improvement from full county information (%)')
    axes[0].set_title('Value of county information by mechanism')
    axes[0].set_xticks([25,50,65]);axes[0].legend(frameon=False,fontsize=9)
    axes[1].set_xticks([0,1],['Direct suppression','Persistent adoption'])
    axes[1].set_ylabel('Infections prevented versus no intervention (%)')
    axes[1].set_title('Matched 50% reference-effect scenario')
    axes[1].set_ylim(0,75)
    axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.text(.07,.015,'Controlled sensitivity: historical epidemic network, synthetic Uniform(0,1) county responsiveness.\n20 assignments; same budget and full-network eligibility. Reference matching uses constant p=0.5.',fontsize=9)
    fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(out/'matched_mechanisms.png',dpi=180)


if __name__=='__main__':main()
