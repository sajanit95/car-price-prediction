# removes personal data before the dataset is used or shared
# seller names and ad URLs -> hashed ids, phone numbers in descriptions -> masked
import hashlib
import re
import sys

import pandas as pd

PHONE = re.compile(r"(?:\+?94|0)[\s-]?7\d[\s-]?\d{3}[\s-]?\d{4}")


def short_hash(value, prefix):
    return prefix + hashlib.sha256(str(value).encode()).hexdigest()[:10]


def anonymise(df):
    df = df.copy()
    df["Seller_name"] = df["Seller_name"].map(lambda v: short_hash(v, "seller_"))
    df["Post_URL"] = df["Post_URL"].map(lambda v: short_hash(v, "ad_"))
    df["Description"] = df["Description"].astype(str).str.replace(PHONE, "[phone removed]", regex=True)
    return df


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/dataset_vehicles.csv"
    raw = pd.read_csv(src)
    phones = raw["Description"].astype(str).str.contains(PHONE).sum()
    anonymise(raw).to_csv("data/dataset_vehicles.csv", index=False)
    print(f"Anonymised {len(raw)} rows: seller names and URLs hashed, {phones} phone numbers masked")
