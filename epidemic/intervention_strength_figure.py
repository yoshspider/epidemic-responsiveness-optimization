"""Scientific comparison of prespecified intervention dose sensitivities."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('outputs/intervention_strength')
    data=pd.read_csv(out/'summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    colors={'aggregate':'#8c98a6','oracle':'#147d80'}
    part=data[data.scenario=='random_all_network']
    for policy,label in [('aggregate','Average responsiveness'),('oracle','Exact county responsiveness')]:
        table=part[part.policy==policy].sort_values('effectiveness')
        axes[0].plot(table.effectiveness,table.prevention_percent,'o-',label=label,color=colors[policy])
    axes[0].set_ylabel('Infections prevented versus no intervention (%)')
    axes[0].set_title('Random responsiveness; all counties eligible')
    axes[0].legend(frameon=False,fontsize=9)
    for scenario,label,color in [('random_all_network','Random p: all counties','#147d80'),('random_common71','Random p: survey cohort','#5d5fa5'),('acceptance','Vaccine acceptance','#c47c22'),('recommendation','Recommendation receptiveness','#ab4561')]:
        table=data[(data.scenario==scenario)&(data.policy=='oracle')].sort_values('effectiveness')
        axes[1].plot(table.effectiveness,table.gain_percent,'o-',label=label,color=color)
    axes[1].set_ylabel('Improvement over average-informed policy (%)')
    axes[1].set_title('Value of county-specific responsiveness')
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.set_xscale('log');ax.set_xticks([1,3,10],['1 (original)','3','10'])
        ax.set_xlabel('Intervention effectiveness multiplier κ')
        ax.spines[['top','right']].set_visible(False)
    fig.text(.08,.015,'Sensitivity analysis, not calibrated effects. Fixed per-capita budget within each cohort.\nExpected-value SIR; random scenarios average 20 assignments. Survey proxies cover 71 eligible counties.',fontsize=9)
    fig.tight_layout(rect=[0,.10,1,1]);fig.savefig(out/'intervention_strength.png',dpi=180)


if __name__=='__main__':main()
