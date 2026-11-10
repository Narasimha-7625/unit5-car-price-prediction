"""
Data Preprocessing Module for Indian Used Car Price Prediction (CART).

Responsibilities:
- Load CSV dataset safely using Pandas
- Parse Indian currency formats (₹, Lakh, Crore, Thousand) and unit strings (kmpl, CC, bhp)
- Inspect dimensions, column types, and missing values
- Remove exact duplicates and impossible/invalid vehicle records
- Build a Scikit-learn ColumnTransformer with median/mode imputation and
  unknown-category-safe OneHotEncoder to prevent data leakage.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MULTI_WORD_BRANDS = (
    "Maruti Suzuki",
    "Land Rover",
    "Aston Martin",
    "Rolls Royce",
    "Mercedes Benz",
    "Mercedes-Benz",
    "Force Motors",
    "MG Motor",
)


def parse_indian_price(value: Any) -> float:
    """Convert Indian price strings (e.g., '₹ 5,45,000', '6.25 Lakh', '1.2 Cr') to INR float."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text or text in {"na", "nan", "null", "none", "-"}:
        return np.nan

    cleaned = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return np.nan
    base = float(match.group(0))

    if "crore" in cleaned or re.search(r"\bcr\b", cleaned):
        return base * 10_000_000.0
    if "lakh" in cleaned or "lac" in cleaned:
        return base * 100_000.0
    if "thousand" in cleaned or re.search(r"\bk\b", cleaned):
        return base * 1_000.0
    return base


def parse_numeric_unit(value: Any) -> float:
    """Extract leading floating-point value from unit strings like '21.21 kmpl' or '1197 CC'."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"na", "nan", "null", "none", "-"}:
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else np.nan


def _split_brand_model(full_name: Any) -> Tuple[str, str]:
    """Split combined vehicle 'name' (CarDekho style) into (brand, model)."""
    if pd.isna(full_name):
        return ("Unknown", "Unknown")
    clean = str(full_name).strip()
    for multi in MULTI_WORD_BRANDS:
        if clean.lower().startswith(multi.lower()):
            rest = clean[len(multi) :].strip()
            return (multi, rest or multi)
    parts = clean.split()
    if len(parts) <= 1:
        return (clean or "Unknown", clean or "Unknown")
    return (parts[0], " ".join(parts[1:]))


def _normalize_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Map common Indian used-car CSV column names to a consistent canonical schema."""
    alias_map = {
        "selling_price": [
            "selling_price",
            "price",
            "resale_price",
            "used_price",
            "sellingprice",
            "car_prices_in_rupee",
            "car_price_in_rupee",
            "car_price",
        ],
        "brand": ["brand", "make", "company", "manufacturer"],
        "model": ["model", "car_model", "variant"],
        "name": ["name", "car_name", "vehicle_name", "title"],
        "year": ["year", "model_year", "manufacturing_year", "reg_year", "manufacture", "manufactured_year"],
        "km_driven": ["km_driven", "kms_driven", "kilometers_driven", "kilometres_driven", "odometer"],
        "fuel": ["fuel", "fuel_type"],
        "transmission": ["transmission", "transmission_type", "gearbox"],
        "owner": ["owner", "owner_type", "ownership", "owners"],
        "mileage": ["mileage", "fuel_economy"],
        "engine": ["engine", "engine_cc", "engine_capacity", "cc"],
        "max_power": ["max_power", "power", "bhp", "horsepower"],
        "seats": ["seats", "seating_capacity", "no_of_seats"],
    }

    normalized_lookup = {
        re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_"): col for col in df.columns
    }
    rename_dict: Dict[str, str] = {}
    for canonical, candidates in alias_map.items():
        if canonical in df.columns:
            continue
        for cand in candidates:
            if cand in normalized_lookup:
                rename_dict[normalized_lookup[cand]] = canonical
                break

    return df.rename(columns=rename_dict)


def _normalize_fuel_label(val: Any) -> Any:
    if pd.isna(val):
        return val
    s = str(val).strip()
    lower = s.lower()
    if lower == "cng":
        return "CNG"
    if lower == "lpg":
        return "LPG"
    if lower == "petrol":
        return "Petrol"
    if lower == "diesel":
        return "Diesel"
    if lower in {"electric", "ev"}:
        return "Electric"
    return s


def _normalize_owner_label(val: Any) -> Any:
    if pd.isna(val):
        return val
    s = str(val).strip()
    lower = s.lower()
    if lower == "0" or lower.startswith("0th"):
        return "0th Owner"
    if "first" in lower or lower == "1" or lower.startswith("1st"):
        return "First Owner"
    if "second" in lower or lower == "2" or lower.startswith("2nd"):
        return "Second Owner"
    if "third" in lower or lower == "3" or lower.startswith("3rd"):
        return "Third Owner"
    if "fourth" in lower or "fifth" in lower or lower.startswith("4") or lower.startswith("5"):
        return "Fourth & Above Owner"
    if "test" in lower:
        return "Test Drive Car"
    return s


def load_and_clean_dataset(csv_source: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load CSV via Pandas, audit dimensions/missing values, normalize Indian units,
    deduplicate, and remove invalid vehicle entries without discarding valid records.
    """
    raw_df = pd.read_csv(csv_source)
    if raw_df.empty:
        raise ValueError("The provided CSV dataset is empty.")

    # Drop synthetic Pandas unnamed row-index columns before deduplication
    unnamed_cols = [c for c in raw_df.columns if str(c).strip().lower().startswith("unnamed")]
    working_df = raw_df.drop(columns=unnamed_cols) if unnamed_cols else raw_df.copy()

    df = _normalize_column_aliases(working_df)
    if "selling_price" not in df.columns:
        raise ValueError("Missing required target price column ('selling_price' or 'price').")

    report: Dict[str, Any] = {
        "raw_rows": int(raw_df.shape[0]),
        "raw_columns": int(raw_df.shape[1]),
        "column_dtypes": {col: str(dtype) for col, dtype in raw_df.dtypes.items()},
        "missing_values_before": {col: int(cnt) for col, cnt in raw_df.isna().sum().items()},
        "steps_applied": [],
    }

    # Derive brand & model if only combined 'name' exists
    if "brand" not in df.columns and "name" in df.columns:
        split_pairs = df["name"].apply(_split_brand_model)
        df["brand"] = [p[0] for p in split_pairs]
        df["model"] = [p[1] for p in split_pairs]
        report["steps_applied"].append("Extracted 'brand' and 'model' from combined 'name' column.")

    # Remove exact duplicate rows (excluding row-index column)
    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(before_dedup - len(df))

    if "fuel" in df.columns:
        df["fuel"] = df["fuel"].apply(_normalize_fuel_label)
    if "owner" in df.columns:
        df["owner"] = df["owner"].apply(_normalize_owner_label)

    # Parse target price in INR
    df["selling_price"] = df["selling_price"].apply(parse_indian_price)
    median_raw_price = df["selling_price"].dropna().median()
    if pd.notna(median_raw_price) and 0 < median_raw_price < 250:
        df["selling_price"] = df["selling_price"] * 100_000.0
        report["steps_applied"].append(
            "Detected bare prices < 250 (Lakhs); converted target prices to INR (x 1,00,000)."
        )

    # Parse numeric text-formatted columns
    for num_col in ["year", "km_driven", "engine", "mileage", "max_power", "seats"]:
        if num_col in df.columns:
            df[num_col] = df[num_col].apply(parse_numeric_unit)

    # Identify invalid / physically impossible vehicle records
    current_max_year = dt.datetime.now().year + 1
    valid_mask = df["selling_price"].notna() & (df["selling_price"] > 5000) & (df["selling_price"] <= 5e8)
    if "year" in df.columns:
        valid_mask &= df["year"].isna() | ((df["year"] >= 1985) & (df["year"] <= current_max_year))
    if "km_driven" in df.columns:
        valid_mask &= df["km_driven"].isna() | ((df["km_driven"] >= 0) & (df["km_driven"] <= 2_000_000))
    if "engine" in df.columns:
        valid_mask &= df["engine"].isna() | ((df["engine"] >= 0) & (df["engine"] <= 8000))
    if "mileage" in df.columns:
        valid_mask &= df["mileage"].isna() | ((df["mileage"] >= 0) & (df["mileage"] <= 100))
    if "max_power" in df.columns:
        valid_mask &= df["max_power"].isna() | ((df["max_power"] >= 0) & (df["max_power"] <= 1500))
    if "seats" in df.columns:
        valid_mask &= df["seats"].isna() | ((df["seats"] >= 2) & (df["seats"] <= 14))

    report["invalid_rows_removed"] = int((~valid_mask).sum())
    cleaned_df = df.loc[valid_mask].reset_index(drop=True)

    if len(cleaned_df) < 25:
        raise ValueError(
            f"Only {len(cleaned_df)} valid rows remained after cleaning. At least 25 records are required."
        )

    # Select only legitimate vehicle specification features (prevent target leakage)
    numeric_candidates = ["year", "km_driven", "engine", "mileage", "max_power", "seats"]
    categorical_candidates = ["brand", "model", "fuel", "transmission", "owner"]

    active_numeric: List[str] = [c for c in numeric_candidates if c in cleaned_df.columns]
    active_categorical: List[str] = [c for c in categorical_candidates if c in cleaned_df.columns]

    report["cleaned_rows"] = int(len(cleaned_df))
    report["active_numeric_features"] = active_numeric
    report["active_categorical_features"] = active_categorical
    report["steps_applied"].extend(
        [
            f"Removed {report['duplicates_removed']} duplicate(s) and {report['invalid_rows_removed']} invalid row(s).",
            "Parsed Indian units (kmpl, CC, bhp) into floats; deferred imputation/encoding until after train-test split.",
        ]
    )

    keep_cols = active_categorical + active_numeric + ["selling_price"]
    return cleaned_df[keep_cols], report


def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    """
    Create a Scikit-learn ColumnTransformer pipeline.
    - Numerical features: median imputation (robust to skewed car specs)
    - Categorical features: most-frequent imputation + OneHotEncoder(handle_unknown='ignore')
      so unseen categories during live prediction never raise errors.
    """
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
