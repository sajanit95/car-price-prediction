# validate -> clean -> train & select -> evaluate -> save
# usage: python run_pipeline.py --config config.yaml
import argparse
import json
import logging
import sys

import pandas as pd
import yaml

from src.features import clean
from src.train import run_training
from src.validate import DataValidationError, validate_raw

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pipeline").info


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    log("STEP 1/4  Validate raw data")
    raw = pd.read_csv(cfg["data_path"])
    try:
        log(f"  {validate_raw(raw)}")
    except DataValidationError as err:
        log(f"  Validation FAILED: {err}")
        sys.exit(1)
    log("STEP 2/4  Clean and engineer features")
    df, clean_log = clean(raw)
    log(f"  {json.dumps(clean_log)}")
    log("STEP 3/4  Train, compare, tune and select best model")
    result = run_training(df, cfg, log)
    log(f"STEP 4/4  Saved model version {result['version']} ({result['model']}) to {cfg['artifacts_dir']}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
