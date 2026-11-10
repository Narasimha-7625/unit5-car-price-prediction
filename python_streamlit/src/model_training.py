"""
CART Regression Model Training Module.

Strictly uses Scikit-learn's DecisionTreeRegressor (CART algorithm) inside a
leakage-free Pipeline with GridSearchCV hyperparameter tuning on training data only.
Saves the complete fitted pipeline artifact using Joblib.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_preprocessing import build_preprocessor, load_and_clean_dataset
from src.evaluation import evaluate_regression_model


def train_and_save_cart_model(
    csv_source: Any,
    model_output_path: str = "models/cart_model.joblib",
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Train, tune, evaluate, and persist the CART DecisionTreeRegressor pipeline.
    """
    cleaned_df, prep_report = load_and_clean_dataset(csv_source)

    num_cols = prep_report["active_numeric_features"]
    cat_cols = prep_report["active_categorical_features"]
    feature_cols = cat_cols + num_cols

    X = cleaned_df[feature_cols]
    y = cleaned_df["selling_price"]

    # 80:20 Train-Test Split BEFORE fitting any imputer or encoder (zero data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Strictly CART Regression (DecisionTreeRegressor) — no ensembles or other models
    cart_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", DecisionTreeRegressor(criterion="squared_error", random_state=random_state)),
        ]
    )

    target_var = float(y_train.var()) if len(y_train) > 1 else 0.0
    param_grid = {
        "regressor__max_depth": [4, 6, 8, 10, 14, None],
        "regressor__min_samples_split": [2, 5, 10],
        "regressor__min_samples_leaf": [1, 3, 5],
        "regressor__ccp_alpha": [0.0, round(target_var * 0.002, 2), round(target_var * 0.01, 2)],
    }

    cv_folds = min(5, max(3, len(X_train) // 8))
    grid_search = GridSearchCV(
        estimator=cart_pipeline,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="r2",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline: Pipeline = grid_search.best_estimator_

    # Evaluate strictly on untouched 20% test data
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)
    metrics = evaluate_regression_model(
        y_true=y_test,
        y_pred=y_test_pred,
        y_train_true=y_train,
        y_train_pred=y_train_pred,
    )

    # Extract feature names and importances
    fitted_pre = best_pipeline.named_steps["preprocessor"]
    fitted_tree: DecisionTreeRegressor = best_pipeline.named_steps["regressor"]
    feature_names = list(fitted_pre.get_feature_names_out())

    schema_metadata = _extract_ui_schema(cleaned_df, num_cols, cat_cols)

    artifact = {
        "pipeline": best_pipeline,
        "algorithm": "CART DecisionTreeRegressor",
        "random_state": random_state,
        "train_count": int(len(X_train)),
        "test_count": int(len(X_test)),
        "best_params": grid_search.best_params_,
        "cv_folds": cv_folds,
        "best_cv_r2": float(grid_search.best_score_),
        "metrics": metrics,
        "feature_names": feature_names,
        "feature_importances": fitted_tree.feature_importances_.tolist(),
        "preprocessing_report": prep_report,
        "schema_metadata": schema_metadata,
        "y_test": y_test.tolist(),
        "y_test_pred": y_test_pred.tolist(),
    }

    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    joblib.dump(artifact, model_output_path)
    return artifact


def _extract_ui_schema(
    df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]
) -> Dict[str, Any]:
    """Capture valid categories and numeric bounds so the UI matches the trained model."""
    categories = {
        col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in cat_cols
    }
    numeric_stats = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }
        for col in num_cols
    }
    return {
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "categories": categories,
        "numeric_stats": numeric_stats,
    }


if __name__ == "__main__":
    result = train_and_save_cart_model("data/dataset.csv", "models/cart_model.joblib")
    print("Trained CART model saved to models/cart_model.joblib")
    print("Best Hyperparameters:", result["best_params"])
    print("Untouched Test Metrics:", result["metrics"])
