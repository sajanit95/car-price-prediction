# cleaning + feature engineering (used by notebook, pipeline and app)
import numpy as np
import pandas as pd

CAT_FEATURES = ["brand_model", "Brand", "Condition", "Transmission", "Body", "Fuel", "district"]
NUM_FEATURES = ["age", "mileage_km", "capacity_cc", "km_per_year", "unregistered"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "price"

DROP_COLUMNS = ["Post_URL", "Seller_name", "Seller_type", "Sub_title", "Edition", "Description"]

MIN_VALID_PRICE = 100_000
MAX_KM_PER_YEAR = 80_000
OUTLIER_FACTOR = 3
MIN_GROUP_SIZE = 5


def _to_number(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9]", "", regex=True), errors="coerce")


def parse_raw(df):
    df = df.copy()
    df["price"] = _to_number(df["Price"])
    df["mileage_km"] = _to_number(df["Mileage"])
    df["capacity_cc"] = _to_number(df["Capacity"])
    df["Body"] = df["Body"].replace({"CoupÃ©/Sports": "Coupe/Sports"})
    return df


def remove_duplicates(df):
    df = df.drop_duplicates(subset="Post_URL")
    repost_key = ["Title", "Price", "Year", "Mileage", "Location", "Seller_name", "Description"]
    return df.drop_duplicates(subset=repost_key)


def add_features(df):
    df = df.copy()
    posted_year = pd.to_datetime(df["published_date"]).dt.year if "published_date" in df else 2021
    df["age"] = (posted_year - df["Year"]).clip(lower=0)
    df["brand_model"] = df["Brand"].str.strip() + " " + df["Model"].str.strip()
    df["district"] = df["Location"].astype(str).str.split(",").str[-1].str.strip()
    df["unregistered"] = df["Title"].astype(str).str.contains("unregist", case=False).astype(int)
    return df


def fix_capacity(df):
    df = df.copy()
    ev = df["Fuel"].eq("Electric")
    litres = (~ev) & (df["capacity_cc"] < 10)  # "2 cc" = 2.0L
    df.loc[litres, "capacity_cc"] *= 1000
    typical = df[~ev].groupby("brand_model")["capacity_cc"].transform("median")
    impossible = (~ev) & (df["capacity_cc"] < 600) & (typical >= 600)
    df.loc[impossible, "capacity_cc"] = np.nan
    df.loc[ev, "capacity_cc"] = 0  # no cc for EVs
    return df, {"litres_converted": int(litres.sum()), "set_missing": int(impossible.sum()),
                "electric_set_zero": int(ev.sum())}


def fix_mileage(df):
    df = df.copy()
    km_per_year = df["mileage_km"] / df["age"].clip(lower=1)
    bad = (km_per_year > MAX_KM_PER_YEAR) | ((df["mileage_km"] == 0) & (df["Condition"] == "Used"))
    df.loc[bad, "mileage_km"] = np.nan
    df["km_per_year"] = df["mileage_km"] / df["age"].clip(lower=1)
    return df, {"set_missing": int(bad.sum())}


def clean(df_raw):
    log = {"raw_rows": len(df_raw)}
    df = parse_raw(df_raw)
    df = remove_duplicates(df)
    log["after_duplicates"] = len(df)
    df = df[df["price"] >= MIN_VALID_PRICE]
    log["after_price_floor"] = len(df)
    df = add_features(df)
    df, log["capacity"] = fix_capacity(df)
    df, log["mileage"] = fix_mileage(df)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    return df.reset_index(drop=True), log


def fit_price_reference(train_df):
    # only call this on the training set
    return (train_df.groupby(["brand_model", "Year"])["price"]
            .agg(group_median="median", group_size="size").reset_index())


def flag_price_outliers(df, reference):
    merged = df[["brand_model", "Year", "price"]].merge(reference, on=["brand_model", "Year"], how="left")
    ratio = np.abs(np.log(merged["price"] / merged["group_median"]))
    return ((merged["group_size"] >= MIN_GROUP_SIZE) & (ratio > np.log(OUTLIER_FACTOR))).to_numpy()


def prepare_input(records):
    df = pd.DataFrame(records).copy()
    if "published_date" not in df:
        df["published_date"] = "2021-07-01"
    if "Title" not in df:
        df["Title"] = np.where(df.get("unregistered", 0) == 1, "unregistered", "")
    df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")
    df["capacity_cc"] = pd.to_numeric(df["capacity_cc"], errors="coerce")
    df = add_features(df)
    df["km_per_year"] = df["mileage_km"] / df["age"].clip(lower=1)
    df.loc[df["Fuel"].eq("Electric"), "capacity_cc"] = 0
    return df[FEATURES]
