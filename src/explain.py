# saves SHAP importance and partial dependence plots for the trained model
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay

from src.features import CAT_FEATURES, FEATURES, NUM_FEATURES


def shap_importance(model, X, random_state=42):
    pipe = model.regressor_
    prep, est = pipe.named_steps["preprocess"], pipe.named_steps["model"]
    Xs = X.sample(min(1000, len(X)), random_state=random_state)
    Xt = prep.transform(Xs)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    values = shap.TreeExplainer(est).shap_values(Xt)

    def original(n):
        return next((c for c in sorted(CAT_FEATURES, key=len, reverse=True) if n.startswith(c + "_")), n)
    grouped = pd.DataFrame(values, columns=names).T.groupby([original(n) for n in names]).sum().T
    imp = grouped.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"mean_abs_shap": imp, "typical_price_effect": np.exp(imp) - 1})


def explain(model, X_train, out_dir, random_state=42):
    out = Path(out_dir) / "explain"
    out.mkdir(parents=True, exist_ok=True)
    imp = shap_importance(model, X_train, random_state)
    imp.to_csv(out / "shap_importance.csv")
    ax = imp["typical_price_effect"].sort_values().plot.barh(figsize=(7, 4), color="#8D153A")
    ax.set(title="SHAP: typical price effect per feature", xlabel="effect on price")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout(); plt.savefig(out / "shap_importance.png", dpi=120); plt.close()

    X = X_train[FEATURES].sample(1500, random_state=random_state).astype({c: float for c in NUM_FEATURES})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    PartialDependenceDisplay.from_estimator(model, X, ["age", "mileage_km", "capacity_cc"], kind="average",
                                            percentiles=(0.02, 0.98), grid_resolution=25, ax=ax)
    for a in ax:
        a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
    plt.tight_layout(); plt.savefig(out / "partial_dependence.png", dpi=120); plt.close()
    return imp
