import re
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1G16aFHLtI8VkzDEuFvmVigIBexySzI8a98HoNPmGPBo/edit?usp=drivesdk"
)


st.set_page_config(
    page_title="Google Sheets Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }
    h1, h2, h3 {
        letter-spacing: 0 !important;
    }
    .hero {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 18px 20px;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 70%, #ecfeff 100%);
        margin-bottom: 16px;
    }
    .hero-title {
        font-size: 1.95rem;
        font-weight: 760;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 0.98rem;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }
    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 650;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 760;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding-left: 16px;
        padding-right: 16px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_sheet_id(sheet_url: str) -> str:
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not match:
        raise ValueError("Cannot find the Google Sheet ID in the URL.")
    return match.group(1)


def extract_gid(sheet_url: str) -> str:
    parsed = urlparse(sheet_url)
    query_gid = parse_qs(parsed.query).get("gid", [None])[0]
    if query_gid:
        return query_gid

    fragment_gid = parse_qs(parsed.fragment).get("gid", [None])[0]
    if fragment_gid:
        return fragment_gid

    return "0"


def build_csv_url(sheet_url: str) -> str:
    sheet_id = extract_sheet_id(sheet_url)
    gid = extract_gid(sheet_url)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def compact_number(value: float) -> str:
    if value is None or pd.isna(value):
        return "-"

    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)

    if value >= 1_000_000_000:
        return f"{sign}{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{sign}{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{sign}{value / 1_000:.2f}K"
    if value >= 10:
        return f"{sign}{value:,.0f}"
    return f"{sign}{value:,.2f}"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(col).strip() or f"Column {idx + 1}" for idx, col in enumerate(df.columns)]

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["", "nan", "None", "NaT"]), col] = np.nan

    return df


def infer_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        series = df[col].astype(str).str.strip()
        non_blank = series.replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna()
        if len(non_blank) == 0:
            continue

        cleaned = series.str.replace("$", "", regex=False)
        cleaned = cleaned.str.replace("VND", "", regex=False)
        cleaned = cleaned.str.replace("USD", "", regex=False)
        cleaned = cleaned.str.replace("%", "", regex=False)
        cleaned = cleaned.str.strip()

        # Handle both English decimals (8.68) and Vietnamese decimals (8,68).
        has_decimal_comma = cleaned.str.match(r"^-?\d+,\d+$", na=False)
        cleaned = cleaned.where(~has_decimal_comma, cleaned.str.replace(",", ".", regex=False))
        cleaned = cleaned.where(has_decimal_comma, cleaned.str.replace(",", "", regex=False))
        numeric = pd.to_numeric(cleaned, errors="coerce")
        valid_ratio = numeric.notna().sum() / max(1, len(non_blank))

        if valid_ratio >= 0.72:
            if series.str.contains("%", regex=False, na=False).mean() > 0.3:
                numeric = numeric / 100
            df[col] = numeric

    return df


def infer_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        col_key = str(col).lower()

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            looks_like_date = any(token in col_key for token in ["date", "ngay", "time", "timestamp"])
            if looks_like_date:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.dropna().between(20000, 60000).mean() >= 0.72:
                    df[col] = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
        valid_ratio = parsed.notna().sum() / max(1, len(series))

        if valid_ratio >= 0.72:
            df[col] = parsed

    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_sheet(sheet_url: str) -> pd.DataFrame:
    csv_url = build_csv_url(sheet_url)
    df = pd.read_csv(csv_url)
    df = normalize_columns(df)
    df = infer_numeric_columns(df)
    df = infer_datetime_columns(df)
    return df


def make_empty_state(message: str) -> None:
    st.error(message)
    st.info(
        "Open the Google Sheet, click Share, set General access to "
        "'Anyone with the link', choose Viewer, then rerun the app."
    )


try:
    secret_sheet_url = st.secrets.get("google_sheet_url", DEFAULT_SHEET_URL)
except Exception:
    secret_sheet_url = DEFAULT_SHEET_URL


with st.sidebar:
    st.header("Data Source")
    sheet_url = st.text_input("Google Sheet URL", value=secret_sheet_url)
    st.caption("Use a public view-only Google Sheet link, or put it in Streamlit Secrets.")

    refresh = st.button("Refresh data", use_container_width=True)
    if refresh:
        st.cache_data.clear()

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Google Sheets Performance Dashboard</div>
        <div class="hero-subtitle">
            A live Streamlit dashboard connected to your online spreadsheet.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    data = load_sheet(sheet_url)
except Exception as exc:
    make_empty_state(f"Could not read the Google Sheet. Details: {exc}")
    st.stop()


if data.empty:
    make_empty_state("The spreadsheet was loaded, but no usable rows were found.")
    st.stop()


date_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
identifier_cols = {"stt", "id", "index", "mahs", "ma_hs", "studentid", "student_id"}
numeric_cols = [
    col
    for col in data.columns
    if pd.api.types.is_numeric_dtype(data[col]) and str(col).strip().lower() not in identifier_cols
]
category_cols = [
    col
    for col in data.columns
    if col not in numeric_cols
    and col not in date_cols
    and data[col].nunique(dropna=True) <= max(30, int(len(data) * 0.35))
]

with st.sidebar:
    st.header("Dashboard Controls")

    if numeric_cols:
        primary_metric = st.selectbox("Primary metric", numeric_cols, index=0)
        secondary_metric = st.selectbox(
            "Secondary metric",
            numeric_cols,
            index=min(1, len(numeric_cols) - 1),
        )
    else:
        primary_metric = None
        secondary_metric = None

    date_col = st.selectbox("Date column", ["None"] + date_cols, index=0)
    category_col = st.selectbox("Category column", ["None"] + category_cols, index=0)
    aggregation = st.selectbox("Aggregation", ["sum", "mean", "median", "count"], index=0)


row_count = len(data)
column_count = len(data.columns)
numeric_count = len(numeric_cols)
category_count = len(category_cols)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", f"{row_count:,}")
k2.metric("Columns", f"{column_count:,}")
k3.metric("Numeric Fields", f"{numeric_count:,}")
k4.metric("Category Fields", f"{category_count:,}")

if primary_metric:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Total {primary_metric}", compact_number(data[primary_metric].sum()))
    m2.metric(f"Average {primary_metric}", compact_number(data[primary_metric].mean()))
    m3.metric(f"Minimum {primary_metric}", compact_number(data[primary_metric].min()))
    m4.metric(f"Maximum {primary_metric}", compact_number(data[primary_metric].max()))


st.divider()

overview_tab, analysis_tab, table_tab = st.tabs(["Overview", "Analysis", "Data Table"])

template = "plotly_white"
palette = ["#2563eb", "#0f766e", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"]

with overview_tab:
    if not primary_metric:
        st.warning("No numeric columns were detected. Add at least one numeric column to build charts.")
    else:
        left, right = st.columns([1.45, 1])

        with left:
            if date_col != "None":
                trend = (
                    data.dropna(subset=[date_col])
                    .groupby(pd.Grouper(key=date_col, freq="D"))[primary_metric]
                    .agg(aggregation)
                    .reset_index()
                    .dropna()
                )
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=trend[date_col],
                        y=trend[primary_metric],
                        mode="lines",
                        line=dict(color=palette[0], width=3),
                        fill="tozeroy",
                        fillcolor="rgba(37, 99, 235, 0.12)",
                        name=primary_metric,
                    )
                )
                fig.update_layout(
                    title=f"{primary_metric} Trend",
                    template=template,
                    height=430,
                    hovermode="x unified",
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                rolling = data[primary_metric].reset_index(drop=True).rolling(7, min_periods=1).mean()
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(1, len(data) + 1),
                        y=data[primary_metric],
                        mode="lines",
                        line=dict(color="#94a3b8", width=1),
                        name="raw",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(1, len(data) + 1),
                        y=rolling,
                        mode="lines",
                        line=dict(color=palette[0], width=3),
                        name="rolling mean",
                    )
                )
                fig.update_layout(
                    title=f"{primary_metric} Sequence",
                    template=template,
                    height=430,
                    hovermode="x unified",
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

        with right:
            if category_col != "None":
                bar_data = (
                    data.groupby(category_col, dropna=False)[primary_metric]
                    .agg(aggregation)
                    .reset_index()
                    .sort_values(primary_metric, ascending=True)
                    .tail(12)
                )
                fig = px.bar(
                    bar_data,
                    x=primary_metric,
                    y=category_col,
                    orientation="h",
                    title=f"Top {category_col}",
                    color=primary_metric,
                    color_continuous_scale="Teal",
                    template=template,
                )
                fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.histogram(
                    data,
                    x=primary_metric,
                    nbins=28,
                    title=f"{primary_metric} Distribution",
                    template=template,
                    color_discrete_sequence=[palette[1]],
                )
                fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig, use_container_width=True)

with analysis_tab:
    col_a, col_b = st.columns(2)

    with col_a:
        if primary_metric:
            fig = px.box(
                data,
                y=primary_metric,
                x=None if category_col == "None" else category_col,
                title=f"{primary_metric} Spread",
                template=template,
                color=None if category_col == "None" else category_col,
            )
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if primary_metric and secondary_metric and primary_metric != secondary_metric:
            fig = px.scatter(
                data,
                x=primary_metric,
                y=secondary_metric,
                color=None if category_col == "None" else category_col,
                title=f"{secondary_metric} vs {primary_metric}",
                template=template,
                color_discrete_sequence=palette,
                opacity=0.82,
            )
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Choose two different numeric fields to show a relationship chart.")

    if len(numeric_cols) >= 2:
        corr = data[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix",
            template=template,
        )
        fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

with table_tab:
    st.subheader("Source Data")
    st.dataframe(data, use_container_width=True, hide_index=True)

    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cleaned CSV",
        csv,
        file_name="google_sheet_dashboard_data.csv",
        mime="text/csv",
        use_container_width=False,
    )
