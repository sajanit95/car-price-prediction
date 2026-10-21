import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split

from src.features import FEATURES, TARGET, fit_price_reference, flag_price_outliers
from src.models import SEARCH_SPACES, candidate_estimators, make_pipeline

SCORING = {"mae": "neg_mean_absolute_error", "r2": "r2", "mape": "neg_mean_absolute_percentage_error"}


def split_and_filter(df, cfg, log=print):
    train, test = train_test_split(df, test_size=cfg["test_size"], random_state=cfg["random_state"])
    reference = fit_price_reference(train)
    out_tr, out_te = flag_price_outliers(train, reference), flag_price_outliers(test, reference)
    log(f"Price outliers removed: {out_tr.sum()} train, {out_te.sum()} test")
    return train[~out_tr], test[~out_te]


def compare_models(X, y, cfg, log=print):
    rows = []
    estimators = candidate_estimators(cfg["random_state"])
    cv = KFold(cfg["cv_folds"], shuffle=True, random_state=cfg["random_state"])
    for name in cfg["candidates"]:
        start = time.time()
        res = cross_validate(make_pipeline(estimators[name]), X, y, cv=cv, scoring=SCORING)
        rows.append({"model": name, "cv_mae": -res["test_mae"].mean(), "cv_r2": res["test_r2"].mean(),
                     "cv_mape": -res["test_mape"].mean(), "seconds": time.time() - start})
        log(f"  {name:<14} CV MAE Rs {rows[-1]['cv_mae']:>12,.0f}   R2 {rows[-1]['cv_r2']:.3f}")
    return pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)


def tune_models(names, X, y, cfg, log=print):
    # keep defaults if tuning doesn't beat them on the same folds
    estimators = candidate_estimators(cfg["random_state"])
    cv = KFold(cfg["tune_cv_folds"], shuffle=True, random_state=cfg["random_state"])
    tuned = {}
    for name in names:
        search = RandomizedSearchCV(make_pipeline(estimators[name]), SEARCH_SPACES[name],
                                    n_iter=cfg["tune_iterations"], cv=cv,
                                    scoring="neg_mean_absolute_error", random_state=cfg["random_state"])
        search.fit(X, y)
        default_mae = -cross_validate(make_pipeline(estimators[name]), X, y, cv=cv,
                                      scoring="neg_mean_absolute_error")["test_score"].mean()
        tuned_mae = -search.best_score_
        if tuned_mae < default_mae:
            tuned[name] = {"model": search.best_estimator_, "cv_mae": tuned_mae,
                           "params": search.best_params_, "used": "tuned"}
        else:
            tuned[name] = {"model": make_pipeline(estimators[name]).fit(X, y), "cv_mae": default_mae,
                           "params": {}, "used": "default"}
        log(f"  {name:<14} default CV MAE Rs {default_mae:>11,.0f} | tuned Rs {tuned_mae:>11,.0f}"
            f" -> keep {tuned[name]['used']}")
    return tuned


def evaluate(model, X, y):
    pred = model.predict(X)
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)),
            "mape": float(mean_absolute_percentage_error(y, pred)),
            "median_ape": float(np.median(np.abs(pred - y) / y))}, pred


def save_artifacts(best_name, best_model, train, test_metrics, test_pred, test, comparison, tuned, cfg):
    out = Path(cfg["artifacts_dir"])
    out.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    joblib.dump(best_model, out / "car_price_model.joblib")
    ratio = test[TARGET].to_numpy() / test_pred
    metadata = {
        "version": version, "model": best_name, "features": FEATURES, "test_metrics": test_metrics,
        "settings_used": tuned[best_name]["used"],
        "best_params": {k.replace("regressor__model__", ""): (v if isinstance(v, (int, float, str)) else str(v))
                        for k, v in tuned[best_name]["params"].items()},
        "fair_range_ratio": [float(np.quantile(ratio, 0.10)), float(np.quantile(ratio, 0.90))],
        "options": {
            "brand_models": (train.groupby("Brand")["Model"]
                             .apply(lambda s: sorted(s.str.strip().value_counts().loc[lambda c: c >= 3].index))
                             .to_dict()),
            "Condition": sorted(train["Condition"].dropna().unique()),
            "Transmission": sorted(train["Transmission"].dropna().unique()),
            "Body": sorted(train["Body"].dropna().unique()),
            "Fuel": sorted(train["Fuel"].dropna().unique()),
            "district": train["district"].value_counts().index.tolist(),
            "year_range": [int(train["Year"].min()), int(train["Year"].max())],
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    comparison.to_csv(out / "model_comparison.csv", index=False)
    train[FEATURES].sample(2000, random_state=cfg["random_state"]).to_csv(out / "lime_background.csv", index=False)
    return version


def run_training(df, cfg, log=print):
    train, test = split_and_filter(df, cfg, log)
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    log(f"Comparing {len(cfg['candidates'])} models with {cfg['cv_folds']}-fold CV ...")
    comparison = compare_models(X_tr, y_tr, cfg, log)
    top = comparison["model"].head(cfg["tune_top_k"]).tolist()
    log(f"Tuning top {len(top)}: {top}")
    tuned = tune_models(top, X_tr, y_tr, cfg, log)
    best_name = min(tuned, key=lambda n: tuned[n]["cv_mae"])
    best_model = tuned[best_name]["model"]
    log(f"Selected model: {best_name}")
    test_metrics, test_pred = evaluate(best_model, X_te, y_te)
    log(f"Test MAE Rs {test_metrics['mae']:,.0f} | R2 {test_metrics['r2']:.3f} | "
        f"median error {test_metrics['median_ape']:.1%}")
    comparison["tuned_cv_mae"] = comparison["model"].map({n: t["cv_mae"] for n, t in tuned.items()})
    version = save_artifacts(best_name, best_model, train, test_metrics, test_pred, test, comparison, tuned, cfg)
    if cfg.get("use_mlflow"):
        import mlflow
        with mlflow.start_run(run_name=f"{best_name}-{version}"):
            mlflow.log_params({"model": best_name, **tuned[best_name]["params"]})
            mlflow.log_metrics(test_metrics)
            mlflow.log_artifacts(cfg["artifacts_dir"])
    return {"version": version, "model": best_name, "test_metrics": test_metrics, "comparison": comparison}
