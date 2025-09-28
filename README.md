# Quick EDA Studio (Streamlit)

A zero-setup Streamlit app for quick exploratory data analysis (EDA) using Plotly.

## Run locally

```bash
# (Recommended) create a virtual env
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Features
- CSV upload + sample datasets (Iris, Wine, Diabetes)
- Numeric & categorical summaries
- Missingness bar chart
- Correlation heatmap
- Scatter, Boxplot, Scatter Matrix
- 2D PCA projection
- One-click lightweight HTML report export
