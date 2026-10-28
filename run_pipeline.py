# validate -> clean -> train & select -> evaluate -> save -> explain
# usage: python run_pipeline.py --config config.yaml
import argparse
import json
import logging
import sys

import pandas as pd
import yaml

from src.features import clean
from src.explain import explain
from src.train import run_training
from src.validate import DataValidationError, validate_raw

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pipeline").info


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    log("STEP 1/5  Validate raw data")
    raw = pd.read_csv(cfg["data_path"])
    try:
        log(f"  {validate_raw(raw)}")
    except DataValidationError as err:
        log(f"  Validation FAILED: {err}")
        sys.exit(1)
    log("STEP 2/5  Clean and engineer features")
    df, clean_log = clean(raw)
    log(f"  {json.dumps(clean_log)}")
    log("STEP 3/5  Train, compare, tune and select best model")
    result = run_training(df, cfg, log)
    log(f"STEP 4/5  Saved model version {result['version']} ({result['model']}) to {cfg['artifacts_dir']}/")
    log("STEP 5/5  Explain model (SHAP + partial dependence)")
    imp = explain(result["best_model"], result["X_train"], cfg["artifacts_dir"], cfg["random_state"])
    log(f"  top features: {', '.join(imp.index[:5])} -> {cfg['artifacts_dir']}/explain/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
