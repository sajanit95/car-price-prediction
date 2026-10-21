import pandas as pd

REQUIRED_COLUMNS = ["Title", "Price", "Brand", "Model", "Year", "Condition", "Transmission", "Body",
                    "Fuel", "Capacity", "Mileage", "Location", "Post_URL", "Seller_name",
                    "Description", "published_date"]
MIN_ROWS = 1000
MAX_UNPARSEABLE_SHARE = 0.05


class DataValidationError(Exception):
    pass


def validate_raw(df):
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if len(df) < MIN_ROWS:
        problems.append(f"Only {len(df)} rows (need at least {MIN_ROWS})")
    for col in ["Price", "Mileage", "Capacity"]:
        parsed = pd.to_numeric(df[col].astype(str).str.replace(r"[^0-9]", "", regex=True), errors="coerce")
        share = parsed.isna().mean()
        if share > MAX_UNPARSEABLE_SHARE:
            problems.append(f"{col}: {share:.1%} of values cannot be read as numbers")
    years = pd.to_numeric(df["Year"], errors="coerce")
    if years.isna().mean() > MAX_UNPARSEABLE_SHARE:
        problems.append("Year: too many non-numeric values")
    if problems:
        raise DataValidationError("; ".join(problems))
    return {"rows": len(df), "columns": len(df.columns), "status": "passed"}
