# Used Car Price Prediction System Using CART Regression

## 1. Project Introduction
An academic Machine Learning project demonstrating the **Classification and Regression Trees (CART)** algorithm (`DecisionTreeRegressor` from Scikit-learn) to estimate used car resale prices in **Indian Rupees (₹)** for the Indian automotive market.

## 2. Problem Statement
Determining a fair resale price for a pre-owned car in India involves complex interactions among brand, manufacturing year, odometer reading, fuel type, transmission, engine capacity, mileage, power, and ownership history. Manual quotes are subjective and inconsistent. This system provides a transparent, data-driven machine learning valuation backed by explainable binary decision rules.

## 3. Project Objectives
1. Demonstrate the practical implementation and mathematical working of CART Regression exclusively using `DecisionTreeRegressor`.
2. Build a leakage-free preprocessing pipeline handling Indian currency formats (`₹`, `Lakh`, `Crore`) and unit strings (`kmpl`, `CC`, `bhp`).
3. Optimize CART generalization via cross-validated hyperparameter tuning (`max_depth`, `min_samples_split`, `min_samples_leaf`, `ccp_alpha`).
4. Evaluate test-set performance rigorously using MAE, MSE, RMSE, R², residual analysis, and tree visualization.
5. Deliver an interactive, reliable multi-page web application with CSV upload and live valuation.

## 4. Algorithm Explanation (CART)
- **Strictly CART Regression:** Uses `sklearn.tree.DecisionTreeRegressor(criterion="squared_error")`. Ensemble models (Random Forest, Gradient Boosting, XGBoost, AdaBoost) and Neural Networks are intentionally excluded.
- **Splitting Criterion:** Selects feature thresholds minimizing Sum of Squared Errors (`SSE`) / variance across left and right child nodes.
- **Why not Gini Impurity?** Gini impurity measures class label mixing for Classification Trees; continuous price regression requires variance/squared-error reduction instead.
- **Pruning & Tuning:** Uses `max_depth`, `min_samples_split`, `min_samples_leaf`, and Minimal Cost-Complexity Pruning (`ccp_alpha`) via `GridSearchCV` to prevent overfitting.

## 5. Dataset Requirements & Source Instructions
- **Built-in Demonstration Data:** `data/dataset.csv` is included strictly as a sample demonstration template.
- **Using a Real Kaggle Dataset:** Download an Indian used-car CSV such as Kaggle's *Vehicle Dataset from CarDekho* (`Car details v3.csv`). Required target column: `selling_price` (or `price`), alongside vehicle features (`name` or `brand`/`model`, `year`, `km_driven`, `fuel`, `transmission`, `owner`, `mileage`, `engine`, `max_power`, `seats`). Upload directly via the **Dataset Explorer** page or replace `data/dataset.csv`.

## 6. Technology Stack
- **Language:** Python 3.10+
- **ML & Data:** Scikit-learn, Pandas, NumPy, Joblib
- **Visualization:** Plotly, Matplotlib
- **Web Interface:** Streamlit

## 7. Project Architecture
```text
used-car-price-prediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── dataset.csv
├── models/
│   └── cart_model.joblib
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── model_training.py
    ├── prediction.py
    ├── evaluation.py
    └── visualization.py
```

## 8. Installation Instructions
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 9. Dataset Setup
Place your downloaded Kaggle Indian used-car CSV at `data/dataset.csv` (or keep the bundled sample template and upload your CSV later inside the app).

## 10. Model Training Instructions
```bash
python -m src.model_training
```
This cleans the dataset, performs an 80:20 split, runs `GridSearchCV` on the training set, evaluates on the untouched test set, and saves `models/cart_model.joblib`.

## 11. Application Execution Instructions
```bash
streamlit run app.py
```

## 12. Explanation of All Pages
1. **Home Dashboard:** Project summary, active dataset status, training record count, and key test metrics.
2. **Price Prediction:** Dynamic, validated input form outputting INR & Lakh estimates plus entered specification summary.
3. **Dataset Explorer:** CSV upload & retraining, schema/missing-value inspection, data preview, and market charts.
4. **Model Analytics:** MAE/MSE/RMSE/R², GridSearchCV parameters, Actual vs Predicted, residuals, feature importances, and tree diagram.
5. **About CART:** Theoretical & mathematical explanation of CART splitting, Gini vs MSE, and pruning.
6. **Project Information:** Abstract, problem statement, objectives, methodology, limitations, and future scope.

## 13. Evaluation Metrics
- **MAE:** Average absolute rupee error.
- **MSE:** Average squared error (splitting objective).
- **RMSE:** Square root of MSE in INR.
- **R²:** Fraction of price variance explained (note: high R² does not guarantee exact predictions for every individual car).

## 14. Limitations
- Piecewise-constant leaf predictions from a single decision tree.
- Physical condition, accident history, and city-specific RTO variations are not captured unless present in the CSV.

## 15. Future Enhancements
- City/RTO registration calibration and leaf quantile prediction intervals.
- Batch CSV appraisal export for used-car dealerships.
