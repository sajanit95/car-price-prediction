import json
from pathlib import Path

import joblib
import pandas as pd

from src.features import prepare_input

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
REQUIRED_INPUTS = ["Brand", "Model", "Year", "mileage_km", "capacity_cc", "Fuel",
                   "Transmission", "Body", "Condition", "Location"]


class InputError(ValueError):
    pass


def load_model(artifacts_dir=ARTIFACTS):
    artifacts_dir = Path(artifacts_dir)
    model = joblib.load(artifacts_dir / "car_price_model.joblib")
    metadata = json.loads((artifacts_dir / "metadata.json").read_text())
    return model, metadata


def _check(records):
    for i, rec in enumerate(records):
        missing = [k for k in REQUIRED_INPUTS if k not in rec or rec[k] in (None, "")]
        if missing:
            raise InputError(f"Row {i}: missing {missing}")
        if not 1900 <= int(rec["Year"]) <= 2021:
            raise InputError(f"Row {i}: Year must be between 1900 and 2021")
        if float(rec["mileage_km"]) < 0:
            raise InputError(f"Row {i}: mileage cannot be negative")


def predict(records, model=None, metadata=None):
    if model is None:
        model, metadata = load_model()
    _check(records)
    X = prepare_input(records)
    pred = model.predict(X)
    low, high = metadata["fair_range_ratio"]
    return pd.DataFrame({"predicted_price": pred.round(-3),
                         "fair_low": (pred * low).round(-3),
                         "fair_high": (pred * high).round(-3)})


def verdict(listed_price, row):
    if listed_price < row["fair_low"]:
        return "Below the typical range - check the listing carefully"
    if listed_price > row["fair_high"]:
        return "Above the typical range - likely overpriced"
    return "Within the typical range - fairly priced"


def predict_batch(csv_path, out_path=None, model=None, metadata=None):
    df = pd.read_csv(csv_path)
    result = pd.concat([df.reset_index(drop=True),
                        predict(df.to_dict("records"), model, metadata)], axis=1)
    if "listed_price" in result:
        result["verdict"] = [verdict(p, r) for p, (_, r) in zip(result["listed_price"], result.iterrows())]
    if out_path:
        result.to_csv(out_path, index=False)
    return result
