import io
import base64
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Quick EDA Studio", layout="wide")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data(show_spinner=False)
def example_dataset(name: str) -> pd.DataFrame:
    if name == "Iris (sklearn)":
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = data.frame
        df["target"] = data.target
        return df
    elif name == "Wine (sklearn)":
        from sklearn.datasets import load_wine
        data = load_wine(as_frame=True)
        df = data.frame
        df["target"] = data.target
        return df
    elif name == "Diabetes (sklearn, numeric)":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df
    else:
        raise ValueError("Unknown example dataset")

def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

def to_csv_download(df: pd.DataFrame, name="data.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name=name, mime="text/csv")

def to_html_download(html: str, name="eda_report.html"):
    b = html.encode("utf-8")
    st.download_button("⬇️ Download HTML report", data=b, file_name=name, mime="text/html")

def numeric_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe().T

def categorical_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    cats = get_categorical_cols(df)
    rows = []
    for c in cats:
        vc = df[c].value_counts(dropna=False).head(10)
        rows.append({
            "column": c,
            "n_unique": df[c].nunique(dropna=True),
            "top": vc.index[0] if len(vc) else None,
            "top_count": int(vc.iloc[0]) if len(vc) else None,
            "missing": int(df[c].isna().sum())
        })
    return pd.DataFrame(rows)

def corr_heatmap(df: pd.DataFrame):
    nums = get_numeric_cols(df)
    if len(nums) < 2:
        st.info("Need at least 2 numeric columns for a correlation heatmap.")
        return
    corr = df[nums].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

def kde_distplot(df: pd.DataFrame, cols: List[str]):
    if not cols:
        st.info("Select 1–4 numeric columns to plot distributions.")
        return
    # Use Plotly Figure Factory for kernel density
    data = [df[c].dropna().values for c in cols]
    fig = ff.create_distplot(data, cols, show_hist=True, show_rug=False)
    fig.update_layout(title="Distribution (KDE + Histogram)")
    st.plotly_chart(fig, use_container_width=True)

def scatter_2d(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None):
    fig = px.scatter(df, x=x, y=y, color=hue, title=f"Scatter: {x} vs {y}", opacity=0.8)
    st.plotly_chart(fig, use_container_width=True)

def boxplot(df: pd.DataFrame, x: Optional[str], y: str, color: Optional[str] = None):
    fig = px.box(df, x=x, y=y, color=color, points="suspectedoutliers", title="Boxplot")
    st.plotly_chart(fig, use_container_width=True)

def pair_scatter(df: pd.DataFrame, cols: List[str], hue: Optional[str] = None):
    if len(cols) < 2:
        st.info("Pick at least 2 columns for a scatter matrix.")
        return
    fig = px.scatter_matrix(df, dimensions=cols, color=hue, title="Scatter Matrix")
    st.plotly_chart(fig, use_container_width=True)

def pca_projection(df: pd.DataFrame, cols: List[str], hue: Optional[str] = None):
    if len(cols) < 2:
        st.info("Pick at least 2 numeric columns to run PCA.")
        return
    x = df[cols].dropna().values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=42)
    comp = pca.fit_transform(x)
    plot_df = pd.DataFrame(comp, columns=["PC1", "PC2"])
    if hue and hue in df.columns:
        # Align hue labels by dropping NaNs from numeric subset first
        valid_idx = df[cols].dropna().index
        plot_df[hue] = df.loc[valid_idx, hue].values
    exp = pca.explained_variance_ratio_
    fig = px.scatter(plot_df, x="PC1", y="PC2", color=hue, title=f"PCA (2D) — VarExpl: {exp[0]:.2f}, {exp[1]:.2f}")
    st.plotly_chart(fig, use_container_width=True)

# ---------- UI ----------
with st.sidebar:
    st.title("⚡ Quick EDA Studio")
    st.caption("Upload a CSV or use a sample dataset to explore quickly.")
    src = st.radio("Data source", ["Upload CSV", "Example dataset"], horizontal=True)
    df = None
    if src == "Upload CSV":
        file = st.file_uploader("Upload a CSV file", type=["csv"])
        if file:
            try:
                df = load_csv(file)
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
    else:
        example = st.selectbox("Choose an example", ["Iris (sklearn)", "Wine (sklearn)", "Diabetes (sklearn, numeric)"])
        df = example_dataset(example)

    st.divider()
    st.markdown("### Global settings")
    show_profile = st.checkbox("Show quick profile card", value=True)
    target = None
    if df is not None:
        # allow selecting a target column (optional)
        cols = df.columns.tolist()
        target = st.selectbox("Optional: choose target/label column", ["<none>"] + cols)
        if target == "<none>":
            target = None

if df is None:
    st.markdown("## 👋 Welcome to **Quick EDA Studio**")
    st.write("Use the left sidebar to upload your CSV or pick a sample dataset.")
    st.info("Tip: After loading your data, you'll see summary stats, distributions, correlations, and more.")
    st.stop()

st.markdown("## 📊 Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Numeric", f"{len(get_numeric_cols(df))}")
c4.metric("Categorical", f"{len(get_categorical_cols(df))}")

st.write("### Preview")
st.dataframe(df.head(50), use_container_width=True)
to_csv_download(df, "dataset.csv")

if show_profile:
    st.write("### Quick profile")
    prof1, prof2, prof3 = st.columns(3)
    prof1.write("**Missing cells**")
    prof1.write(int(df.isna().sum().sum()))
    prof2.write("**Duplicate rows**")
    prof2.write(int(df.duplicated().sum()))
    prof3.write("**Memory usage (MB)**")
    prof3.write(round(df.memory_usage(deep=True).sum() / (1024**2), 3))

st.divider()
st.markdown("## 🧮 Summaries")

tabs = st.tabs(["Numeric", "Categorical", "Missingness", "Correlation"])

with tabs[0]:
    nums = get_numeric_cols(df)
    if nums:
        st.dataframe(numeric_summary_table(df[nums]), use_container_width=True)
        st.markdown("#### Distribution (KDE + Hist)")
        sel_kde = st.multiselect("Pick up to 4 numeric columns", nums[:4], max_selections=4)
        kde_distplot(df, sel_kde)
    else:
        st.info("No numeric columns detected.")

with tabs[1]:
    cats = get_categorical_cols(df)
    if cats:
        st.dataframe(categorical_summary_table(df), use_container_width=True)
    else:
        st.info("No categorical columns detected.")

with tabs[2]:
    miss = df.isna().sum().sort_values(ascending=False)
    miss_df = miss[miss > 0].reset_index()
    miss_df.columns = ["column", "missing"]
    if len(miss_df):
        fig = px.bar(miss_df, x="column", y="missing", title="Missing values by column")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values! 🎉")

with tabs[3]:
    corr_heatmap(df)

st.divider()
st.markdown("## 📈 Visual Explorer")

nums = get_numeric_cols(df)
cats = get_categorical_cols(df)
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Scatter")
    if len(nums) >= 2:
        x = st.selectbox("X", nums, index=0, key="scatter_x")
        y = st.selectbox("Y", nums, index=min(1, len(nums)-1), key="scatter_y")
        hue = st.selectbox("Color (optional)", ["<none>"] + df.columns.tolist(), index=0, key="scatter_hue")
        if hue == "<none>":
            hue = None
        scatter_2d(df, x, y, hue)
    else:
        st.info("Need at least 2 numeric columns for scatter.")

with col2:
    st.markdown("#### Boxplot")
    if nums:
        yb = st.selectbox("Y", nums, index=0, key="box_y")
        xb_opt = ["<none>"] + cats + nums
        xb = st.selectbox("Group (x)", xb_opt, index=0, key="box_x")
        if xb == "<none>":
            xb = None
        color = st.selectbox("Color", ["<none>"] + df.columns.tolist(), index=0, key="box_color")
        if color == "<none>":
            color = None
        boxplot(df, xb, yb, color)
    else:
        st.info("No numeric columns for boxplot.")

st.markdown("#### Scatter Matrix")
dims = st.multiselect("Pick 2–6 columns", df.columns.tolist()[:4], max_selections=6)
hue2 = st.selectbox("Color (optional)", ["<none>"] + df.columns.tolist(), index=0, key="pair_hue")
if hue2 == "<none>":
    hue2 = None
pair_scatter(df, dims, hue2)

st.divider()
st.markdown("## 🧭 PCA (2D projection)")
nums_for_pca = st.multiselect("Numeric columns", nums[: min(5, len(nums))])
hue3 = st.selectbox("Color (optional)", ["<none>"] + df.columns.tolist(), index=0, key="pca_hue")
if hue3 == "<none>":
    hue3 = None
pca_projection(df, nums_for_pca, hue3)

st.divider()
st.markdown("## 📝 One-click HTML Report")
with st.expander("Configure and export a lightweight HTML report"):
    include_sections = st.multiselect(
        "Sections to include",
        ["Overview", "Numeric summary", "Categorical summary", "Missingness", "Correlation"],
        default=["Overview", "Numeric summary", "Missingness", "Correlation"],
    )
    if st.button("Generate report"):
        parts = []
        if "Overview" in include_sections:
            parts.append(f"<h2>Overview</h2><p>Rows: {len(df)}, Columns: {df.shape[1]}</p>")
        if "Numeric summary" in include_sections and len(nums):
            parts.append("<h2>Numeric summary</h2>" + numeric_summary_table(df[nums]).to_html())
        if "Categorical summary" in include_sections and len(cats):
            parts.append("<h2>Categorical summary</h2>" + categorical_summary_table(df).to_html())
        if "Missingness" in include_sections:
            miss_table = df.isna().sum().to_frame("missing").T
            parts.append("<h2>Missingness</h2>" + miss_table.to_html())
        if "Correlation" in include_sections and len(nums) >= 2:
            parts.append("<h2>Correlation (numeric)</h2>" + df[nums].corr().to_html())
        html = f"""<!doctype html><html><head><meta charset='utf-8'><title>EDA Report</title>
        <style>body{{font-family:system-ui,Segoe UI,Arial;padding:24px;}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:6px}}</style>
        </head><body><h1>EDA Report</h1>{''.join(parts)}</body></html>"""
        to_html_download(html, "eda_report.html")

st.caption("Built with ❤️ using Streamlit + Plotly | Quick EDA Studio")
