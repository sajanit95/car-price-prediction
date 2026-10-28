import pandas as pd
import pytest

from src.features import clean, prepare_input, FEATURES
from src.validate import DataValidationError, validate_raw

RAW = pd.read_csv("data/dataset_vehicles.csv", nrows=3000)


def test_validation_passes_on_real_data():
    assert validate_raw(RAW)["status"] == "passed"


def test_validation_rejects_missing_columns():
    with pytest.raises(DataValidationError):
        validate_raw(RAW.drop(columns=["Price"]))


def test_cleaning_produces_numeric_target_and_no_bad_prices():
    df, _ = clean(RAW)
    assert df["price"].dtype.kind in "if"
    assert (df["price"] >= 100_000).all()
    assert "Post_URL" not in df


def test_prepare_input_returns_model_features():
    car = dict(Brand="Toyota", Model="Aqua", Year=2014, mileage_km=90000, capacity_cc=1500, Fuel="Hybrid",
               Transmission="Automatic", Body="Hatchback", Condition="Used", Location="Colombo")
    X = prepare_input([car])
    assert list(X.columns) == FEATURES
    assert X["age"].iloc[0] == 7


def test_no_personal_data_left():
    from anonymise_data import PHONE
    df = pd.read_csv("data/dataset_vehicles.csv")
    assert not df["Description"].astype(str).str.contains(PHONE).any()
    assert df["Seller_name"].str.startswith("seller_").all()


def test_main_model_is_not_covered_in_lectures():
    import yaml
    from src.models import COVERED_IN_LECTURES
    assert yaml.safe_load(open("config.yaml"))["main_model"] not in COVERED_IN_LECTURES
