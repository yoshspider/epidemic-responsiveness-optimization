"""Separate sentiment from posting volume and quantify state-cluster uncertainty."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .prediction import predict


def main():
    folder=Path('outputs/tbcov_prediction')
    frame=pd.read_csv(folder/'features_targets.csv',dtype={'fips':str})
    sets={'population':['log_population'],
          'population_activity':['log_population','log_post_count'],
          'sentiment_only':['mean_sentiment','positive_share','negative_share'],
          'combined':['log_population','log_post_count','mean_sentiment','positive_share','negative_share']}
    rows=[]
    uncertainty=[]
    for model in ('ridge','forest'):
        predictions,folds=predict(frame,sets,5,model)
        predictions.to_csv(folder/(model+'_ablation_predictions.csv'),index=False)
        folds.to_csv(folder/(model+'_ablation_fold_metrics.csv'),index=False)
        for name in sets:
            values=predictions[name+'_predicted'].to_numpy()
            true=frame.responsiveness.to_numpy()
            error=values-true
            rows.append({'model':model,'features':name,'mae':abs(error).mean(),'rmse':np.sqrt((error**2).mean()),
                         'r2':1-(error**2).sum()/((true-true.mean())**2).sum()})
        for baseline in ('population','population_activity'):
            improvement=abs(predictions[baseline+'_predicted'].to_numpy()-frame.responsiveness.to_numpy())-abs(predictions.combined_predicted.to_numpy()-frame.responsiveness.to_numpy())
            grouped=pd.DataFrame({'state':frame.geo_group,'improvement':improvement}).groupby('state').improvement.agg(['sum','count'])
            rng=np.random.default_rng(521)
            samples=rng.integers(0,len(grouped),(5000,len(grouped)))
            means=grouped['sum'].to_numpy()[samples].sum(1)/grouped['count'].to_numpy()[samples].sum(1)
            low,high=np.quantile(means,[.025,.975])
            uncertainty.append({'model':model,'baseline':baseline,'comparison':'combined',
                                'mae_reduction':improvement.mean(),'state_cluster_ci_low':low,'state_cluster_ci_high':high,
                                'states':len(grouped),'bootstrap_repetitions':5000})
    pd.DataFrame(rows).to_csv(folder/'ablation_summary.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(folder/'ablation_cluster_intervals.csv',index=False)
    (folder/'ablation_method.json').write_text(json.dumps({'purpose':'Attribute gains to sentiment beyond population and posting volume',
        'validation':'Same fixed five-fold state-blocked validation; no tuning on fold outcomes',
        'intervals':'Bootstrap paired out-of-fold absolute-error differences by state, exploratory and unadjusted; does not refit models inside bootstrap'},indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(uncertainty).to_string(index=False))


if __name__=='__main__':
    main()
