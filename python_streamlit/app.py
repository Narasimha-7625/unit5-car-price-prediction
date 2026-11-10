"""
Used Car Price Prediction System Using CART Regression (DecisionTreeRegressor).
Complete 6-page Streamlit Web Application.
"""

from __future__ import annotations

import datetime as dt
import os

import pandas as pd
import streamlit as st
from sklearn.tree import export_text

from src.data_preprocessing import load_and_clean_dataset
from src.evaluation import get_metric_explanations
from src.model_training import train_and_save_cart_model
from src.prediction import (
    format_inr,
    format_lakh,
    load_saved_artifact,
    predict_car_price,
)
from src.visualization import (
    plot_actual_vs_predicted,
    plot_cart_tree_matplotlib,
    plot_feature_importance,
    plot_km_vs_price,
    plot_price_by_fuel,
    plot_price_by_transmission,
    plot_price_distribution,
    plot_residual_distribution,
    plot_year_vs_price,
)

MODEL_PATH = "models/cart_model.joblib"
DEFAULT_DATA_PATH = "data/dataset.csv"

st.set_page_config(
    page_title="Used Car Price Prediction — CART Regression",
    page_icon="🚗",
    layout="wide",
)

# Custom CSS for clean automotive dashboard styling
st.markdown(
    """
    <style>
    .main { background-color: #F8FAFC; }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
    }
    .price-banner {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 12px;
        padding: 20px 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_or_train_artifact():
    """Load persisted CART model without retraining on every refresh."""
    if os.path.exists(MODEL_PATH):
        return load_saved_artifact(MODEL_PATH)
    if os.path.exists(DEFAULT_DATA_PATH):
        return train_and_save_cart_model(DEFAULT_DATA_PATH, MODEL_PATH)
    raise FileNotFoundError("Dataset file not found. Upload a CSV in Dataset Explorer.")


if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = DEFAULT_DATA_PATH
if "dataset_source_label" not in st.session_state:
    st.session_state.dataset_source_label = "car_price_238cca5c.csv (Indian Used-Car Dataset, 5,512 rows)"

st.sidebar.title("CART Car Valuation")
st.sidebar.caption("Unit 5 ML Project · DecisionTreeRegressor Only")
page = st.sidebar.radio(
    "Navigation",
    [
        "1. Home Dashboard",
        "2. Price Prediction",
        "3. Dataset Explorer",
        "4. Model Analytics",
        "5. About CART",
        "6. Project Information",
    ],
)

try:
    artifact = get_or_train_artifact()
except Exception as exc:
    st.error(f"Initialization error: {exc}")
    st.stop()


# PAGE 1: HOME DASHBOARD
if page == "1. Home Dashboard":
    st.title("Used Car Price Prediction System Using CART Regression")
    st.markdown(
        "Estimate Indian used-car resale prices in **Indian Rupees (₹)** using a single, "
        "transparent **Classification and Regression Trees (`DecisionTreeRegressor`)** model."
    )
    st.info(
        f"**Active Dataset Source:** {st.session_state.dataset_source_label}. "
        "Upload a real Kaggle Indian used-car CSV in **Dataset Explorer** at any time to retrain."
    )

    m = artifact["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Records (80%)", f"{artifact['train_count']:,}")
    c2.metric("Algorithm", "CART Regressor")
    c3.metric("Test R² Score", f"{m['r2']:.4f}")
    c4.metric("Test MAE", format_lakh(m["mae"]))

    st.subheader("How the Application Works")
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown(
        "**01. Data Preprocessing**\n\nCleans Indian currency/unit strings (`kmpl`, `CC`, `bhp`, `Lakh`), "
        "removes duplicates and invalid records, and avoids leakage by fitting encoders only on training data."
    )
    col_b.markdown(
        "**02. GridSearchCV CART Tuning**\n\nTunes `max_depth`, `min_samples_split`, `min_samples_leaf`, "
        "and `ccp_alpha` via cross-validation on the training split."
    )
    col_c.markdown(
        "**03. Interactive Valuation**\n\nLoads the persisted `Joblib` pipeline to predict resale value "
        "and explain the exact decision tree rules."
    )


# PAGE 2: PRICE PREDICTION
elif page == "2. Price Prediction":
    st.title("Interactive Used Car Price Prediction")
    schema = artifact["schema_metadata"]
    cat_cols = schema["categorical_features"]
    num_cols = schema["numeric_features"]

    st.caption(
        f"Form fields dynamically match the {len(cat_cols) + len(num_cols)} active features in the trained CART model."
    )

    user_inputs = {}
    with st.form("prediction_form"):
        cols = st.columns(2)
        idx = 0
        for col in cat_cols:
            options = schema["categories"].get(col, [])
            with cols[idx % 2]:
                user_inputs[col] = st.selectbox(col.replace("_", " ").title(), options)
            idx += 1

        current_year = dt.datetime.now().year
        for col in num_cols:
            stats = schema["numeric_stats"][col]
            with cols[idx % 2]:
                if col == "year":
                    user_inputs[col] = st.number_input(
                        "Manufacturing Year",
                        min_value=1985,
                        max_value=current_year,
                        value=int(stats["median"]),
                        step=1,
                    )
                elif col == "seats":
                    user_inputs[col] = st.number_input(
                        "Number of Seats",
                        min_value=2,
                        max_value=12,
                        value=int(stats["median"]),
                        step=1,
                    )
                elif col == "km_driven":
                    user_inputs[col] = st.number_input(
                        "Kilometres Driven",
                        min_value=0,
                        max_value=1_500_000,
                        value=int(stats["median"]),
                        step=1000,
                    )
                else:
                    user_inputs[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=0.0,
                        max_value=10_000.0,
                        value=float(round(stats["median"], 1)),
                    )
            idx += 1

        submitted = st.form_submit_button("PREDICT PRICE", type="primary")

    if submitted:
        try:
            res = predict_car_price(artifact, user_inputs)
            st.markdown(
                f"""
                <div class="price-banner">
                    <div style="font-size:13px;color:#1E40AF;font-weight:600;">ESTIMATED RESALE PRICE</div>
                    <div style="font-size:32px;font-weight:700;color:#0F172A;">{res['formatted_inr']}</div>
                    <div style="font-size:15px;color:#1E40AF;font-weight:600;">({res['formatted_lakh']})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.warning(
                "This is a machine learning estimate derived from historical dataset averages, "
                "not a guaranteed selling price."
            )
            st.subheader("Entered Vehicle Specifications")
            st.json(res["validated_inputs"])
        except ValueError as err:
            st.error(f"Validation error: {err}")


# PAGE 3: DATASET EXPLORER
elif page == "3. Dataset Explorer":
    st.title("Dataset Explorer & CSV Upload")
    with st.expander("Kaggle Dataset Download Guide", expanded=False):
        st.markdown(
            "1. Download an Indian used-car CSV from Kaggle (e.g., *Car details v3.csv* / CarDekho dataset).\n"
            "2. Ensure it contains `selling_price` (or `price`) plus vehicle features (`year`, `km_driven`, `fuel`, `transmission`, `brand`/`name`, etc.).\n"
            "3. Upload the CSV below and click **Replace Dataset & Retrain CART Model**."
        )

    uploaded = st.file_uploader("Upload Used Car CSV Dataset", type=["csv"])
    if uploaded is not None and st.button("Replace Dataset & Retrain CART Model", type="primary"):
        try:
            os.makedirs("data", exist_ok=True)
            uploaded_path = "data/uploaded_dataset.csv"
            with open(uploaded_path, "wb") as f:
                f.write(uploaded.getbuffer())
            new_artifact = train_and_save_cart_model(uploaded_path, MODEL_PATH)
            st.session_state.dataset_path = uploaded_path
            st.session_state.dataset_source_label = f"Uploaded CSV ({uploaded.name})"
            st.cache_resource.clear()
            st.success(
                f"Retrained CART model on {new_artifact['preprocessing_report']['cleaned_rows']} cleaned records!"
            )
            st.rerun()
        except Exception as err:
            st.error(f"Failed to process uploaded CSV: {err}")

    try:
        cleaned_df, prep_report = load_and_clean_dataset(st.session_state.dataset_path)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Rows", prep_report["raw_rows"])
        c2.metric("Cleaned Rows", prep_report["cleaned_rows"])
        c3.metric("Duplicates Removed", prep_report["duplicates_removed"])
        c4.metric("Invalid Rows Removed", prep_report["invalid_rows_removed"])

        st.subheader("Dataset Preview")
        st.dataframe(cleaned_df.head(50), use_container_width=True)

        st.subheader("Missing Values & Data Types (Raw CSV)")
        profile_df = pd.DataFrame(
            {
                "Column": list(prep_report["column_dtypes"].keys()),
                "Dtype": list(prep_report["column_dtypes"].values()),
                "Missing Count": list(prep_report["missing_values_before"].values()),
            }
        )
        st.dataframe(profile_df, use_container_width=True)

        st.subheader("Exploratory Visualizations")
        st.plotly_chart(plot_price_distribution(cleaned_df), use_container_width=True)
        col1, col2 = st.columns(2)
        fig_fuel = plot_price_by_fuel(cleaned_df)
        if fig_fuel:
            col1.plotly_chart(fig_fuel, use_container_width=True)
        fig_trans = plot_price_by_transmission(cleaned_df)
        if fig_trans:
            col2.plotly_chart(fig_trans, use_container_width=True)
        fig_yr = plot_year_vs_price(cleaned_df)
        if fig_yr:
            col1.plotly_chart(fig_yr, use_container_width=True)
        fig_km = plot_km_vs_price(cleaned_df)
        if fig_km:
            col2.plotly_chart(fig_km, use_container_width=True)
    except Exception as err:
        st.error(f"Error inspecting dataset: {err}")


# PAGE 4: MODEL ANALYTICS
elif page == "4. Model Analytics":
    st.title("CART Model Analytics & Test Evaluation")
    m = artifact["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", format_inr(m["mae"]))
    c2.metric("MSE", f"{m['mse']:,.0f}")
    c3.metric("RMSE", format_inr(m["rmse"]))
    c4.metric("Test R²", f"{m['r2']:.4f}")

    st.warning(
        "Note: A high R² score explains aggregate dataset variance and does NOT guarantee "
        "accurate predictions for every individual car."
    )

    explanations = get_metric_explanations()
    with st.expander("What Do These Metrics Mean?"):
        for k, v in explanations.items():
            st.markdown(f"- **{k}:** {v}")

    st.subheader("Best Hyperparameters & Split Summary")
    st.json(
        {
            "train_records": artifact["train_count"],
            "test_records": artifact["test_count"],
            "cv_folds": artifact["cv_folds"],
            "best_cv_r2": round(artifact["best_cv_r2"], 4),
            "best_params": artifact["best_params"],
        }
    )

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        plot_actual_vs_predicted(artifact["y_test"], artifact["y_test_pred"]),
        use_container_width=True,
    )
    col2.plotly_chart(
        plot_residual_distribution(artifact["y_test"], artifact["y_test_pred"]),
        use_container_width=True,
    )

    st.plotly_chart(
        plot_feature_importance(artifact["feature_names"], artifact["feature_importances"]),
        use_container_width=True,
    )

    depth_view = st.slider("Tree Visualization Depth", min_value=2, max_value=5, value=3)
    st.pyplot(plot_cart_tree_matplotlib(artifact, max_depth=depth_view))


# PAGE 5: ABOUT CART
elif page == "5. About CART":
    st.title("About the CART Regression Algorithm")
    st.markdown(
        """
        ### 1. What is CART?
        **Classification and Regression Trees (CART)** is a binary recursive partitioning algorithm (Breiman et al., 1984).
        In this project, we strictly use `DecisionTreeRegressor` to predict continuous car resale prices.

        ### 2. Why CART for Used Car Price Prediction?
        Car prices depend on non-linear thresholds (e.g., manufacturing year × fuel type × engine displacement).
        CART captures these interactions natively and produces transparent split rules.

        ### 3. Classification vs Regression Trees
        - **Classification Trees** predict discrete categories using **Gini Impurity** or Entropy and output a majority class.
        - **Regression Trees** predict continuous numbers using **Squared-Error Reduction (MSE)** and output the leaf mean.

        ### 4. Why Gini Impurity Is Not Used Here
        Gini impurity ($G = 1 - \\sum p_k^2$) requires discrete class probabilities $p_k$. Since car prices are continuous,
        CART regression minimizes squared-error variance instead.

        ### 5. Squared-Error Split Math
        At node $t$, $SSE(t) = \\sum_{i \\in t} (y_i - \\bar{y}_t)^2$. CART chooses feature $j$ and threshold $s$ maximizing:
        $$\\Delta SSE = SSE(t) - \\left(SSE(t_L) + SSE(t_R)\\right)$$

        ### 6. Hyperparameters, Overfitting & Pruning
        - `max_depth`, `min_samples_split`, and `min_samples_leaf` prevent memorizing noisy training cars (overfitting)
          while avoiding overly shallow trees (underfitting).
        - Cost-complexity pruning (`ccp_alpha`) penalizes unnecessary leaves: $R_\\alpha(T) = R(T) + \\alpha |T|$.
        """
    )

    st.subheader("Actual Trained Tree Rules (Text Export)")
    fitted_tree = artifact["pipeline"].named_steps["regressor"]
    clean_names = [f.replace("num__", "").replace("cat__", "") for f in artifact["feature_names"]]
    st.code(export_text(fitted_tree, feature_names=clean_names, max_depth=3), language="text")


# PAGE 6: PROJECT INFORMATION
elif page == "6. Project Information":
    st.title("Project Information")
    st.markdown(
        """
        ### Abstract & Problem Statement
        This academic Machine Learning project builds an interpretable used-car valuation platform for the Indian market
        using **CART Regression (`DecisionTreeRegressor`)**, addressing inconsistent manual valuations with reproducible data preprocessing and tuned decision trees.

        ### Objectives & Methodology
        - Preprocess Indian vehicle CSV datasets cleanly without data leakage.
        - Tune CART hyperparameters via 5-fold `GridSearchCV`.
        - Evaluate with MAE, MSE, RMSE, and R² on an untouched 20% test split.
        - Persist the trained pipeline using `Joblib` and deploy an interactive 6-page Streamlit web app.

        ### Limitations & Future Enhancements
        - Single trees produce piecewise-constant leaf estimates and exclude unrecorded factors like physical dents or city RTO taxes.
        - Future versions can incorporate city-level registration features and quantile prediction intervals.
        """
    )
