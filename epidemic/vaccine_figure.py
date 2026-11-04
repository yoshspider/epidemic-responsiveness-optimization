"""Static comparison of the matched-cohort proxy experiment."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('outputs/vaccine_experiment')
    data=pd.read_csv(out/'summary.csv')
    data=data[(data['mode']=='stochastic') & (data.fraction==.1)]
    fig,ax=plt.subplots(figsize=(8,4.5))
    colors=['#197c80','#d17b32']
    for index,metric in enumerate(['acceptance','recommendation']):
        rows=data[data.metric==metric].set_index('policy')
        for offset,policy in [(-.16,'oracle'),(.16,'predicted')]:
            row=rows.loc[policy]
            ax.bar(index+offset,row.gain_vs_baseline/1000,width=.28,color=colors[offset>0],label=('Current county survey' if policy=='oracle' else 'Previous-month survey') if index==0 else None)
            ax.errorbar(index+offset,row.gain_vs_baseline/1000,yerr=[[(row.gain_vs_baseline-row.ci_low)/1000],[(row.ci_high-row.gain_vs_baseline)/1000]],color='black',capsize=4)
    ax.set_xticks([0,1],['Vaccine acceptance','Receptiveness to government\nhealth recommendations'])
    ax.set_ylabel('Additional modeled infections prevented (thousands)')
    ax.set_title('Do county attitudes improve allocation?')
    ax.axhline(0,color='black',linewidth=.7)
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.12,.015,'Baseline knows current average responsiveness. Same cohort and budget.\nMarch–April 2021 counterfactual SIR; intervals reflect epidemic noise only.',fontsize=9)
    fig.tight_layout(rect=[0,.09,1,1]);fig.savefig(out/'proxy_allocation_comparison.png',dpi=180)


if __name__=='__main__':main()
