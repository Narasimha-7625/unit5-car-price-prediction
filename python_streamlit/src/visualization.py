"""
Visualization Module for Used Car Price Prediction (CART).

Generates all 8 required data-driven visualizations using Plotly and Matplotlib:
1. Used car price distribution
2. Price distribution by fuel type
3. Price distribution by transmission
4. Manufacturing year vs selling price
5. Kilometres driven vs selling price
6. Actual vs predicted prices
7. Feature importance
8. CART decision tree structure
"""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import plot_tree


def plot_price_distribution(df: pd.DataFrame):
    fig = px.histogram(
        df,
        x="selling_price",
        nbins=20,
        title="Used Car Selling Price Distribution (INR)",
        labels={"selling_price": "Selling Price (INR)"},
        color_discrete_sequence=["#1E40AF"],
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_price_by_fuel(df: pd.DataFrame):
    if "fuel" not in df.columns:
        return None
    fig = px.box(
        df,
        x="fuel",
        y="selling_price",
        color="fuel",
        title="Selling Price Distribution by Fuel Type",
        labels={"fuel": "Fuel Type", "selling_price": "Selling Price (INR)"},
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_price_by_transmission(df: pd.DataFrame):
    if "transmission" not in df.columns:
        return None
    fig = px.box(
        df,
        x="transmission",
        y="selling_price",
        color="transmission",
        title="Selling Price Distribution by Transmission",
        labels={"transmission": "Transmission", "selling_price": "Selling Price (INR)"},
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_year_vs_price(df: pd.DataFrame):
    if "year" not in df.columns:
        return None
    color_col = "fuel" if "fuel" in df.columns else None
    fig = px.scatter(
        df,
        x="year",
        y="selling_price",
        color=color_col,
        title="Manufacturing Year vs Selling Price",
        labels={"year": "Manufacturing Year", "selling_price": "Selling Price (INR)"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_km_vs_price(df: pd.DataFrame):
    if "km_driven" not in df.columns:
        return None
    color_col = "transmission" if "transmission" in df.columns else None
    fig = px.scatter(
        df,
        x="km_driven",
        y="selling_price",
        color=color_col,
        title="Kilometres Driven vs Selling Price",
        labels={"km_driven": "Kilometres Driven", "selling_price": "Selling Price (INR)"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_actual_vs_predicted(y_test: List[float], y_pred: List[float]):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            name="Test Cars",
            marker=dict(color="#1E40AF", size=8, opacity=0.8),
        )
    )
    lo = min(min(y_test), min(y_pred))
    hi = max(max(y_test), max(y_pred))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Ideal Fit (y = x)",
            line=dict(color="#0F172A", dash="dash"),
        )
    )
    fig.update_layout(
        title="Actual vs Predicted Car Prices (Untouched Test Split)",
        xaxis_title="Actual Price (INR)",
        yaxis_title="Predicted Price (INR)",
        template="plotly_white",
    )
    return fig


def plot_residual_distribution(y_test: List[float], y_pred: List[float]):
    residuals = np.array(y_test) - np.array(y_pred)
    fig = px.histogram(
        x=residuals,
        nbins=15,
        title="Prediction Error (Residual) Distribution",
        labels={"x": "Residual = Actual - Predicted (INR)"},
        color_discrete_sequence=["#0F172A"],
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_feature_importance(feature_names: List[str], importances: List[float]):
    grouped: Dict[str, float] = {}
    for name, imp in zip(feature_names, importances):
        clean = name.replace("num__", "").replace("cat__", "")
        base = clean.split("_")[0] if name.startswith("cat__") else clean
        grouped[base] = grouped.get(base, 0.0) + float(imp)

    imp_df = (
        pd.DataFrame({"Feature": list(grouped.keys()), "Importance": list(grouped.values())})
        .sort_values("Importance", ascending=True)
    )
    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="CART Feature Importance (Normalized Variance Reduction)",
        color_discrete_sequence=["#1E40AF"],
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_cart_tree_matplotlib(artifact: Dict[str, Any], max_depth: int = 3):
    fitted_tree = artifact["pipeline"].named_steps["regressor"]
    feature_names = [
        f.replace("num__", "").replace("cat__", "") for f in artifact["feature_names"]
    ]
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        fitted_tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"Trained CART Decision Tree (Top {max_depth} Levels)")
    plt.tight_layout()
    return fig
