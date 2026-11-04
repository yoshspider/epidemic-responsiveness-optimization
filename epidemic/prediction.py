"""Geographic out-of-fold prediction; features must predate the target survey."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .data import fips


def predict(frame, feature_sets, folds=5, model="ridge", test_start=None):
    frame = frame.copy()
    frame["fips"] = fips(frame.fips)
    if test_start is None and frame.fips.duplicated().any():
        raise ValueError("Repeated county-month observations require --test-start temporal holdout")
    target = frame.responsiveness.to_numpy(float)
    if not np.isfinite(target).all() or np.any((target<0)|(target>1)):
        raise ValueError("Invalid target")
    dates = pd.to_datetime(frame.feature_end, errors="raise")
    targets = pd.to_datetime(frame.target_start, errors="raise")
    if dates.isna().any() or targets.isna().any() or (dates >= targets).any():
        raise ValueError("Every feature window must end before its target window")
    if "target_end" in frame:
        target_ends=pd.to_datetime(frame.target_end,errors="raise")
        if target_ends.isna().any() or (target_ends<targets).any():
            raise ValueError("Invalid target_end")
    else:
        target_ends=targets
        if test_start is not None:
            raise ValueError("Temporal holdout requires target_end to purge overlapping target windows")
    if frame.duplicated(["fips","target_start"]).any():
        raise ValueError("Duplicate county target windows")
    if (frame.groupby("fips").geo_group.nunique()>1).any():
        raise ValueError("A county must have one geographic holdout group")
    if frame.geo_group.isna().any() or frame.geo_group.nunique() < folds:
        raise ValueError("Insufficient geographic holdout groups")
    output = frame[["fips", "responsiveness", "geo_group", "target_start"]].copy()
    metrics = []
    forbidden = {"responsiveness", "fips", "geo_group", "feature_end", "target_start"}
    for name, columns in feature_sets.items():
        if not columns or forbidden.intersection(columns):
            raise ValueError("Invalid feature set or target leakage")
        x = frame[columns].apply(pd.to_numeric, errors="raise")
        predictions = np.full(len(frame),np.nan)
        for fold, (train, test) in enumerate(GroupKFold(folds).split(x, target, frame.geo_group)):
            if test_start is not None:
                cutoff=pd.Timestamp(test_start)
                train=train[(target_ends.iloc[train]<cutoff).to_numpy()]
                test=test[(targets.iloc[test]>=cutoff).to_numpy()]
                if len(train)<2 or len(test)<2:
                    raise ValueError("Insufficient pre-cutoff training or post-cutoff test rows in a fold")
            estimator = Ridge(alpha=10) if model == "ridge" else RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)
            pipeline = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(), estimator)
            pipeline.fit(x.iloc[train], target[train])
            values = np.clip(pipeline.predict(x.iloc[test]), 0, 1)
            predictions[test] = values
            metrics.append({"features": name, "fold": fold, "mae": mean_absolute_error(target[test], values),
                            "rmse": np.sqrt(mean_squared_error(target[test], values)), "r2": r2_score(target[test], values)})
        output[name + "_predicted"] = predictions
    return output, pd.DataFrame(metrics)


def main():
    import json
    p = argparse.ArgumentParser()
    p.add_argument("table", type=Path)
    p.add_argument("feature_sets", type=Path, help="JSON mapping demographics/sentiment/combined to column lists")
    p.add_argument("--output", type=Path, default=Path("outputs/prediction"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--model", choices=["ridge", "forest"], default="ridge")
    p.add_argument("--test-start",help="Future evaluation cutoff for county-month panels; target_end required")
    a = p.parse_args()
    predictions, metrics = predict(pd.read_csv(a.table, dtype={"fips":str}), json.loads(a.feature_sets.read_text()), a.folds, a.model,a.test_start)
    a.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(a.output/"out_of_fold_predictions.csv", index=False)
    metrics.to_csv(a.output/"metrics.csv", index=False)


if __name__ == "__main__":
    main()
