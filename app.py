import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.inference import load_model, predict_price
from src.train import DATA_PATH, METADATA_PATH

st.set_page_config(page_title="Laptop Price Studio", page_icon="💻", layout="wide")
st.title("Laptop Price Studio")
st.caption("A machine-learning estimate based on the repository's historical laptop dataset.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def get_model():
    return load_model()

data = load_data()
metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else None

overview = st.columns(3)
overview[0].metric("Dataset records", f"{len(data):,}")
overview[1].metric("Brands represented", str(data["Company"].nunique()))
overview[2].metric("Model", metadata["model_name"] if metadata else "Not trained")

st.divider()
predict_tab, explore_tab, model_tab = st.tabs(["Price estimate", "Dataset explorer", "Model report"])

with predict_tab:
    st.subheader("Configure a laptop")
    with st.form("laptop_configuration"):
        left, right = st.columns(2)
        with left:
            company = st.selectbox("Brand", sorted(data["Company"].dropna().unique()))
            type_name = st.selectbox("Laptop category", sorted(data["TypeName"].dropna().unique()))
            ram = st.selectbox("RAM", sorted(data["Ram"].str.extract(r"(\d+)")[0].dropna().astype(int).unique()))
            weight = st.number_input("Weight (kg)", min_value=0.4, max_value=6.0, value=1.6, step=0.1)
            inches = st.number_input("Screen size (inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
            resolution = st.selectbox("Screen resolution", ["1920x1080", "1366x768", "1600x900", "2560x1600", "3840x2160", "2560x1440", "3200x1800"])
        with right:
            cpu = st.selectbox("CPU", sorted(data["Cpu"].dropna().unique()))
            gpu = st.selectbox("GPU", sorted(data["Gpu"].dropna().unique()))
            memory = st.selectbox("Storage configuration", sorted(data["Memory"].dropna().unique()))
            os_name = st.selectbox("Operating system", sorted(data["OpSys"].dropna().unique()))
        submitted = st.form_submit_button("Estimate laptop price", use_container_width=True)

    if submitted:
        laptop = {
            "Company": company, "TypeName": type_name, "Inches": inches,
            "ScreenResolution": resolution, "Cpu": cpu, "Ram": f"{ram}GB",
            "Memory": memory, "Gpu": gpu, "OpSys": os_name, "Weight": f"{weight}kg",
        }
        try:
            estimate = predict_price(get_model(), laptop)
            st.success(f"Estimated dataset price: {estimate:,.2f}")
            st.caption("This is a model estimate, not a live retail quote. The dataset's original price units are retained.")
        except FileNotFoundError as exc:
            st.error(str(exc))

with explore_tab:
    st.subheader("Dataset overview")
    c1, c2 = st.columns(2)
    with c1:
        counts = data["Company"].value_counts().head(12).sort_values()
        st.bar_chart(counts)
        st.caption("Most represented laptop brands")
    with c2:
        fig, ax = plt.subplots()
        ax.hist(data["Price"].dropna(), bins=30)
        ax.set_xlabel("Price (dataset units)")
        ax.set_ylabel("Laptop count")
        st.pyplot(fig)
        plt.close(fig)
    st.dataframe(data.head(20), use_container_width=True)

with model_tab:
    st.subheader("Evaluation report")
    if metadata:
        metric_cols = st.columns(3)
        metric_cols[0].metric("Holdout MAE", f'{metadata["holdout_metrics"]["mae"]:,.2f}')
        metric_cols[1].metric("Holdout RMSE", f'{metadata["holdout_metrics"]["rmse"]:,.2f}')
        metric_cols[2].metric("Holdout R²", f'{metadata["holdout_metrics"]["r2"]:.3f}')
        st.write("Cross-validation RMSE comparison")
        cv = pd.DataFrame(metadata["cross_validation"]).T.rename_axis("Model")
        st.dataframe(cv, use_container_width=True)
        st.caption("Metrics are measured on a held-out split from this dataset; they do not establish current market-price accuracy.")
    else:
        st.info("Train the model first with: python -m src.train")
