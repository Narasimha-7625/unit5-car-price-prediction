"""
Prediction Module for Used Car Price Prediction (CART).

Loads the persisted Joblib pipeline without retraining on refresh, validates user inputs,
transforms inputs identically to training, and formats prices in Indian Rupees (₹ / lakh).
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd


def load_saved_artifact(model_path: str = "models/cart_model.joblib") -> Dict[str, Any]:
    """Load persisted CART pipeline artifact from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Saved CART model file not found. Please train the model first."
        )
    return joblib.load(model_path)


def format_inr(amount: float) -> str:
    """Format an amount using Indian comma grouping (e.g., ₹5,45,000)."""
    val = max(0, int(round(amount)))
    s = str(val)
    if len(s) <= 3:
        return f"₹{s}"
    last_three = s[-3:]
    rest = s[:-3]
    parts: List[str] = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return f"₹{','.join(parts)},{last_three}"


def format_lakh(amount: float) -> str:
    """Format an amount in Indian Lakhs or Crores (e.g., ₹5.45 lakh)."""
    safe = max(0.0, float(amount))
    if safe >= 10_000_000:
        return f"₹{safe / 10_000_000:.2f} crore"
    return f"₹{safe / 100_000:.2f} lakh"


def validate_user_inputs(
    inputs: Dict[str, Any], schema_metadata: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """Validate user inputs against physical Indian vehicle bounds and active schema."""
    errors: List[str] = []
    validated: Dict[str, Any] = {}
    current_year = dt.datetime.now().year

    for col in schema_metadata.get("categorical_features", []):
        val = str(inputs.get(col, "")).strip()
        if not val:
            errors.append(f"'{col}' cannot be empty.")
        else:
            validated[col] = val

    for col in schema_metadata.get("numeric_features", []):
        raw = inputs.get(col)
        try:
            num = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{col}' must be a valid number.")
            continue

        if col == "year" and (num < 1985 or num > current_year or int(num) != num):
            errors.append(f"Manufacturing year must be an integer between 1985 and {current_year}.")
        elif col == "km_driven" and (num < 0 or num > 1_500_000):
            errors.append("Kilometres driven cannot be negative or exceed 15,00,000 km.")
        elif col == "engine" and (num < 0 or num > 8000):
            errors.append("Engine capacity must be between 0 (EV) and 8000 CC.")
        elif col == "mileage" and (num < 0 or num > 80):
            errors.append("Mileage must be between 0 and 80 kmpl.")
        elif col == "max_power" and (num <= 0 or num > 1200):
            errors.append("Maximum power must be between 1 and 1200 bhp.")
        elif col == "seats" and (num < 2 or num > 12 or int(num) != num):
            errors.append("Number of seats must be an integer between 2 and 12.")
        elif num < 0:
            errors.append(f"'{col}' cannot be negative.")
        else:
            validated[col] = int(num) if col in {"year", "seats"} else num

    return validated, errors


def predict_car_price(
    artifact: Dict[str, Any], user_inputs: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate inputs, apply identical trained pipeline transforms, and predict price."""
    schema = artifact["schema_metadata"]
    validated, errors = validate_user_inputs(user_inputs, schema)
    if errors:
        raise ValueError(" ".join(errors))

    feature_order = schema["categorical_features"] + schema["numeric_features"]
    input_df = pd.DataFrame([{k: validated[k] for k in feature_order}])

    pipeline = artifact["pipeline"]
    raw_pred = float(pipeline.predict(input_df)[0])
    predicted_inr = max(25_000.0, raw_pred)

    return {
        "predicted_inr": predicted_inr,
        "formatted_inr": format_inr(predicted_inr),
        "formatted_lakh": format_lakh(predicted_inr),
        "validated_inputs": validated,
    }
