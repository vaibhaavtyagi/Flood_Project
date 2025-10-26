"""
Streamlit Flood Data Explorer & Quick Model Builder
- Upload a CSV dataset (or an .ipynb) and explore data
- Do quick EDA: preview, missing values, histograms, correlation
- Train a quick ML model (RandomForest) for classification or regression
- Download trained model (.pkl)

Save this file as `streamlit_app.py` and run:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

requirements.txt (example):
streamlit
pandas
numpy
scikit-learn
matplotlib
joblib
nbformat

"""

import io
import json
import tempfile
import base64

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
import joblib
import nbformat

st.set_page_config(page_title="Flood Data Explorer", layout="wide")

st.title("🌊 Flood Data Explorer — Streamlit Quick App")
st.write("Upload a CSV dataset (or a Jupyter .ipynb) to explore, run quick EDA, and build a simple model.")

# Sidebar controls
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV or .ipynb", type=["csv", "ipynb"], accept_multiple_files=False)
preview_rows = st.sidebar.slider("Preview rows", min_value=3, max_value=50, value=10)

if uploaded_file is None:
    st.info("Please upload a CSV file (or a Jupyter notebook) to get started. Example: flood dataset CSV.")
    st.stop()

file_name = uploaded_file.name

# Helper: read CSV
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

# Helper: parse ipynb and extract any embedded CSV-like attachments or display content
def parse_ipynb(file) -> dict:
    try:
        nb = nbformat.reads(file.read().decode("utf-8"), as_version=4)
    except Exception as e:
        return {"error": str(e)}

    cells = []
    for i, c in enumerate(nb.cells):
        cells.append({"cell_index": i, "cell_type": c.cell_type, "source": c.source})
    return {"cells": cells}

# Main: handle file types
if file_name.endswith(".csv"):
    try:
        df = load_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.header("Data Preview")
    st.write(f"**File:** {file_name} — **Rows:** {df.shape[0]} — **Columns:** {df.shape[1]}")
    st.dataframe(df.head(preview_rows))

    with st.expander("Show full dataframe info"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    # Basic EDA
    st.header("Exploratory Data Analysis (Quick)")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Missing values")
        miss = df.isnull().sum().sort_values(ascending=False)
        st.dataframe(miss[miss > 0])

        st.subheader("Numeric summary")
        st.dataframe(df.select_dtypes(include=np.number).describe().T)

    with col2:
        st.subheader("Column types")
        dtypes = pd.DataFrame(df.dtypes, columns=["dtype"]) 
        st.dataframe(dtypes)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        st.subheader("Histograms (select column)")
        col = st.selectbox("Choose numeric column for histogram", options=numeric_cols)
        bins = st.slider("Bins", 5, 200, 30)
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=bins)
        ax.set_title(f"Histogram: {col}")
        st.pyplot(fig)

        if len(numeric_cols) >= 2:
            st.subheader("Correlation heatmap (numeric)")
            corr = df[numeric_cols].corr()
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            im = ax2.imshow(corr, interpolation='nearest')
            ax2.set_xticks(range(len(numeric_cols)))
            ax2.set_xticklabels(numeric_cols, rotation=45, ha='right')
            ax2.set_yticks(range(len(numeric_cols)))
            ax2.set_yticklabels(numeric_cols)
            st.pyplot(fig2)

    # Quick ML model builder
    st.header("Quick Model Builder")
    st.write("Use a simple RandomForest to get a baseline. Select target and features.")

    all_columns = df.columns.tolist()
    target = st.selectbox("Select target column", options=[None] + all_columns)

    if target:
        default_features = [c for c in all_columns if c != target and df[c].dtype in ["int64", "float64"]][:5]
        features = st.multiselect("Select feature columns (numeric recommended)", options=[c for c in all_columns if c != target], default=default_features)

        test_size = st.slider("Test set proportion", 0.05, 0.5, 0.2)
        random_state = st.number_input("Random seed", value=42)

        if st.button("Train model"):
            # Prepare data
            X = df[features].copy()
            y = df[target].copy()

            # Basic preprocessing: drop rows with NA in X or y
            data = pd.concat([X, y], axis=1).dropna()
            X = data[features]
            y = data[target]

            # Determine task type
            is_classification = pd.api.types.is_integer_dtype(y) or y.nunique() <= 20

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=random_state)

            with st.spinner("Training..."):
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            st.subheader("Model results")
            if is_classification:
                acc = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: **{acc:.4f}**")
                st.text(classification_report(y_test, y_pred))
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"MSE: **{mse:.4f}**")
                st.write(f"R²: **{r2:.4f}**")

            # Feature importances
            try:
                fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
                st.subheader("Feature importances")
                st.dataframe(fi)

                fig3, ax3 = plt.subplots()
                fi.plot.bar(ax=ax3)
                st.pyplot(fig3)
            except Exception:
                st.write("Feature importances are not available for this model.")

            # Offer download
            b = io.BytesIO()
            joblib.dump(model, b)
            b.seek(0)

            st.download_button("Download trained model (.pkl)", data=b, file_name="rf_model.pkl", mime="application/octet-stream")
            st.header("📅 Flood Prediction for Future Dates")

            if "date" in df.columns:
                st.write("Select a base date to predict flood conditions for upcoming days.")
                today = pd.Timestamp.today()
                if today.year < 2019:
                    today = pd.Timestamp("2019-01-01")
                elif today.year > 2025:
                    today = pd.Timestamp("2025-12-31")
                st.write(f"Using current date for prediction: {today.date()}")
                input_date = today
            
                if st.button("Predict Flood After 5–7 Days"):
                    try:
                        future_dates = [input_date + pd.Timedelta(days=d) for d in [5, 7]]
                        st.write("🕒 Predicting flood for future dates:", future_dates)
                        last_features = df[features].iloc[-1:].copy()
                        preds = {}
                        for d in future_dates:
                            preds[str(d.date())] = model.predict(last_features)[0]
                        st.subheader("🌧 Flood Prediction Results")
                        results_df = pd.DataFrame({
                            "Date": preds.keys(),
                            "Predicted Flood Value": preds.values()
                        })
                        st.dataframe(results_df)
                        fig, ax = plt.subplots()
                        ax.plot(results_df["Date"], results_df["Predicted Flood Value"], marker='o')
                        ax.set_title("Predicted Flood Severity (Next 5–7 Days)")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Prediction Value")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
            else:
                st.info("No 'date' column found in your dataset. Please ensure your dataset includes a date column for time-based prediction.")
            

else:
    # .ipynb handling: show notebook cells and try to find CSV attachments
    info = parse_ipynb(uploaded_file)
    if info.get("error"):
        st.error(f"Failed to parse notebook: {info['error']}")
        st.stop()

    st.header("Uploaded Jupyter Notebook (.ipynb)")
    st.write(f"**File:** {file_name} — contains {len(info['cells'])} cells")

    for c in info['cells']:
        if c['cell_type'] == 'markdown':
            with st.expander(f"Markdown cell {c['cell_index']}"):
                st.markdown(c['source'])
        else:
            with st.expander(f"Code cell {c['cell_index']}"):
                st.code(c['source'], language='python')

    st.info("If your notebook includes a CSV used in the analysis, please upload that CSV separately to use the data explorer / model builder.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit — modify the file to add custom visualizations or model pipelines.")


