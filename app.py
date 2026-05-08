import re
from html import escape
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1G16aFHLtI8VkzDEuFvmVigIBexySzI8a98HoNPmGPBo/edit#gid=1185619496"
)


st.set_page_config(
    page_title="Dashboard Học Sinh",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1480px;
    }
    h1, h2, h3 {
        letter-spacing: 0 !important;
    }
    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid #dbeafe;
        border-radius: 8px;
        padding: 24px 26px;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(239,246,255,0.94) 58%, rgba(204,251,241,0.78) 100%);
        margin-bottom: 18px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 820;
        color: #0f172a;
        margin-bottom: 5px;
    }
    .hero-subtitle {
        color: #475569;
        font-size: 1.02rem;
        max-width: 820px;
    }
    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-top: 15px;
        padding: 8px 12px;
        border-radius: 8px;
        color: #0f766e;
        background: rgba(240, 253, 250, 0.92);
        border: 1px solid rgba(20, 184, 166, 0.25);
        font-size: 0.88rem;
        font-weight: 720;
    }
    .section-title {
        margin: 8px 0 12px;
        color: #0f172a;
        font-size: 1.1rem;
        font-weight: 780;
    }
    .metric-card {
        min-height: 138px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 17px 18px;
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.06);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.86rem;
        font-weight: 740;
        text-transform: uppercase;
        letter-spacing: 0;
    }
    .metric-value {
        color: #0f172a;
        font-size: 2.05rem;
        font-weight: 860;
        line-height: 1.05;
        margin-top: 8px;
    }
    .metric-subtext {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 10px;
    }
    .metric-good {
        border-top: 4px solid #0f766e;
    }
    .metric-warn {
        border-top: 4px solid #f59e0b;
    }
    .metric-risk {
        border-top: 4px solid #dc2626;
    }
    .metric-blue {
        border-top: 4px solid #2563eb;
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
        font-weight: 650;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] {
        background: #f8fafc;
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

    return ""


def build_csv_url(sheet_url: str) -> str:
    sheet_id = extract_sheet_id(sheet_url)
    gid = extract_gid(sheet_url)
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    if gid:
        csv_url = f"{csv_url}&gid={gid}"
    return csv_url


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
        looks_like_date_name = any(token in col_key for token in ["date", "ngay", "time", "timestamp"])

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if looks_like_date_name:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.dropna().between(20000, 60000).mean() >= 0.72:
                    df[col] = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        sample = series.astype(str).head(20)
        looks_like_date_value = sample.str.contains(
            r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}",
            regex=True,
            na=False,
        ).mean() >= 0.5

        if not (looks_like_date_name or looks_like_date_value):
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
    st.header("Nguồn Dữ Liệu")
    sheet_url = st.text_input("Đường dẫn Google Sheet", value=secret_sheet_url, key="sheet_url_v5")
    st.caption("Dùng Google Sheet công khai quyền xem hoặc cấu hình trong Streamlit Secrets.")

    refresh = st.button("Làm mới dữ liệu", width="stretch")
    if refresh:
        st.cache_data.clear()

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Dashboard Phân Tích Học Tập</div>
        <div class="hero-subtitle">
            Kết nối trực tiếp Google Sheets, tự động đọc bảng điểm và trực quan hóa dữ liệu học sinh.
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
    and data[col].nunique(dropna=True) <= max(40, int(len(data) * 0.42))
]


def find_col(candidates: list[str], columns: list[str]) -> str | None:
    lowered = {str(col).strip().lower(): col for col in columns}

    for candidate in candidates:
        match = lowered.get(candidate.strip().lower())
        if match is not None:
            return match

    for candidate in candidates:
        key = candidate.strip().lower()
        for col in columns:
            if key in str(col).strip().lower():
                return col

    return None


def sorted_options(series: pd.Series) -> list[str]:
    return sorted([str(value) for value in series.dropna().unique()])


def safe_rate(part: int | float, total: int | float) -> float:
    if not total:
        return 0.0
    return float(part) / float(total)


def as_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def metric_card(label: str, value: str, subtext: str, tone: str = "blue") -> None:
    st.markdown(
        f"""
        <div class="metric-card metric-{escape(tone)}">
            <div>
                <div class="metric-label">{escape(label)}</div>
                <div class="metric-value">{escape(value)}</div>
            </div>
            <div class="metric-subtext">{escape(subtext)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


plot_config = {
    "displayModeBar": False,
    "responsive": True,
}
template = "plotly_white"
palette = ["#2563eb", "#0f766e", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#be123c", "#4338ca"]
risk_colors = {"Thap": "#0f766e", "Vua": "#f59e0b", "Cao": "#dc2626"}
status_colors = {"On dinh": "#0f766e", "Theo doi": "#f59e0b", "Can can thiep": "#dc2626"}
grade_colors = {
    "Xuat sac": "#0f766e",
    "Gioi": "#2563eb",
    "Kha": "#0891b2",
    "Trung binh": "#f59e0b",
    "Can ho tro": "#dc2626",
}


def polish(fig: go.Figure, height: int = 380, title_size: int = 18) -> go.Figure:
    fig.update_layout(
        template=template,
        height=height,
        margin=dict(l=18, r=18, t=58, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        colorway=palette,
        title=dict(font=dict(size=title_size, color="#0f172a"), x=0.02, xanchor="left"),
        font=dict(family="Segoe UI, Arial, sans-serif", color="#334155", size=13),
        hoverlabel=dict(bgcolor="#0f172a", font_color="#ffffff", bordercolor="#0f172a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False, linecolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False, linecolor="#e5e7eb")
    return fig


score_col = find_col(["DiemTrungBinh", "DiemTB", "Average", "Score"], data.columns.tolist())
attendance_col = find_col(["ChuyenCan", "Attendance"], data.columns.tolist())
progress_col = find_col(["TienBo", "Progress"], data.columns.tolist())
class_col = find_col(["Lop", "Class"], data.columns.tolist())
grade_col = find_col(["XepLoai", "Grade"], data.columns.tolist())
status_col = find_col(["TrangThai", "Status"], data.columns.tolist())
risk_col = find_col(["MucRuiRo", "Risk"], data.columns.tolist())
student_col = find_col(["HoTen", "Student", "Name"], data.columns.tolist())
gender_col = find_col(["GioiTinh", "Gender"], data.columns.tolist())
group_col = find_col(["NhomHocTap", "Group"], data.columns.tolist())
date_col = find_col(["NgayCapNhat", "Date", "Ngay"], date_cols) if date_cols else None
late_col = find_col(["SoLanDiTre", "Late"], data.columns.tolist())
violation_col = find_col(["SoLanViPham", "Violation"], data.columns.tolist())

test_cols = [
    col
    for col in ["DiemKT1", "DiemKT2", "DiemGiuaKy", "DiemCuoiKy", "DiemDuAn"]
    if col in data.columns and pd.api.types.is_numeric_dtype(data[col])
]
primary_metric = score_col or (numeric_cols[0] if numeric_cols else None)
secondary_metric = attendance_col or (numeric_cols[1] if len(numeric_cols) > 1 else primary_metric)

with st.sidebar:
    st.header("Bộ Lọc")

    filtered = data.copy()

    if class_col:
        selected_classes = st.multiselect(
            "Lớp",
            sorted_options(data[class_col]),
            default=sorted_options(data[class_col]),
            key="class_filter_v4",
        )
        filtered = filtered[filtered[class_col].astype(str).isin(selected_classes)]

    if grade_col:
        selected_grades = st.multiselect(
            "Xếp loại",
            sorted_options(data[grade_col]),
            default=sorted_options(data[grade_col]),
            key="grade_filter_v4",
        )
        filtered = filtered[filtered[grade_col].astype(str).isin(selected_grades)]

    if status_col:
        selected_statuses = st.multiselect(
            "Trạng thái",
            sorted_options(data[status_col]),
            default=sorted_options(data[status_col]),
            key="status_filter_v4",
        )
        filtered = filtered[filtered[status_col].astype(str).isin(selected_statuses)]

    if risk_col:
        selected_risks = st.multiselect(
            "Mức rủi ro",
            sorted_options(data[risk_col]),
            default=sorted_options(data[risk_col]),
            key="risk_filter_v4",
        )
        filtered = filtered[filtered[risk_col].astype(str).isin(selected_risks)]

    if gender_col:
        selected_genders = st.multiselect(
            "Giới tính",
            sorted_options(data[gender_col]),
            default=sorted_options(data[gender_col]),
            key="gender_filter_v4",
        )
        filtered = filtered[filtered[gender_col].astype(str).isin(selected_genders)]

    st.header("Tùy Chỉnh Góc Nhìn")
    st.caption("Phiên bản giao diện v5")
    if numeric_cols:
        default_primary = numeric_cols.index(primary_metric) if primary_metric in numeric_cols else 0
        primary_metric = st.selectbox("Chỉ số chính", numeric_cols, index=default_primary, key="main_metric_v4")

        default_secondary = numeric_cols.index(secondary_metric) if secondary_metric in numeric_cols else 0
        secondary_metric = st.selectbox(
            "So sánh với",
            numeric_cols,
            index=default_secondary,
            key="secondary_metric_v4",
        )

if filtered.empty:
    st.warning("No rows match the selected filters.")
    st.stop()


student_count = len(filtered)
avg_score = filtered[score_col].mean() if score_col else np.nan
avg_attendance = filtered[attendance_col].mean() if attendance_col else np.nan
avg_progress = filtered[progress_col].mean() if progress_col else np.nan
excellent_count = int((filtered[score_col] >= 8).sum()) if score_col else 0
high_risk_count = int((filtered[risk_col].astype(str) == "Cao").sum()) if risk_col else 0
watch_count = int(filtered[status_col].astype(str).isin(["Theo doi", "Can can thiep"]).sum()) if status_col else 0

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">Trung Tâm Điều Hành Học Tập</div>
        <div class="hero-subtitle">
            Theo dõi điểm số, chuyên cần, tiến bộ, xếp loại và tín hiệu rủi ro từ Google Sheets theo thời gian thực.
        </div>
        <div class="hero-chip">Đang đọc: {student_count:,} học sinh / {len(data.columns):,} cột dữ liệu</div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    metric_card("Học sinh", f"{student_count:,}", "Số bản ghi sau khi lọc", "blue")
with m2:
    metric_card("Điểm TB", "-" if pd.isna(avg_score) else f"{avg_score:.2f}", "Điểm tổng hợp có trọng số", "good")
with m3:
    metric_card("Tỷ lệ khá giỏi", as_percent(safe_rate(excellent_count, student_count)), "Điểm từ 8.0 trở lên", "blue")
with m4:
    metric_card("Chuyên cần TB", "-" if pd.isna(avg_attendance) else f"{avg_attendance:.1f}%", "Sức khỏe tham gia lớp học", "good")
with m5:
    metric_card("Cần chú ý", f"{watch_count:,}", f"Rủi ro cao: {high_risk_count:,}", "risk" if watch_count else "warn")

st.divider()

executive_tab, student_tab, analytics_tab, table_tab = st.tabs(
    ["Tổng Quan Điều Hành", "Góc Nhìn Học Sinh", "Phân Tích Sâu", "Bảng Dữ Liệu"]
)

with executive_tab:
    st.markdown('<div class="section-title">Tổng Quan Hiệu Suất Học Tập</div>', unsafe_allow_html=True)

    top_left, top_right = st.columns([1.55, 1])

    with top_left:
        if score_col and date_col:
            trend_source = filtered.dropna(subset=[date_col, score_col]).copy()
            if class_col:
                trend = (
                    trend_source.groupby([pd.Grouper(key=date_col, freq="D"), class_col])[score_col]
                    .mean()
                    .reset_index()
                    .dropna()
                )
                fig = px.line(
                    trend,
                    x=date_col,
                    y=score_col,
                    color=class_col,
                    markers=True,
                    title="Xu hướng điểm trung bình theo lớp",
                    color_discrete_sequence=palette,
                )
            else:
                trend = trend_source.groupby(pd.Grouper(key=date_col, freq="D"))[score_col].mean().reset_index()
                fig = px.area(trend, x=date_col, y=score_col, title="Xu hướng điểm trung bình", color_discrete_sequence=[palette[0]])
            fig.update_traces(line=dict(width=3), marker=dict(size=7))
            fig.update_yaxes(range=[max(0, min(10, float(filtered[score_col].min()) - 0.5)), 10])
            st.plotly_chart(polish(fig, 430), width="stretch", config=plot_config)
        elif primary_metric:
            rolling = filtered[primary_metric].reset_index(drop=True).rolling(5, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(1, len(filtered) + 1), y=filtered[primary_metric], mode="lines", name="Raw", line=dict(color="#94a3b8", width=1)))
            fig.add_trace(go.Scatter(x=np.arange(1, len(filtered) + 1), y=rolling, mode="lines", name="Rolling mean", line=dict(color=palette[0], width=4)))
            fig.update_layout(title=f"{primary_metric} Sequence")
            st.plotly_chart(polish(fig, 430), width="stretch", config=plot_config)

    with top_right:
        if risk_col:
            risk_counts = filtered[risk_col].value_counts().rename_axis(risk_col).reset_index(name="Số học sinh")
            fig = px.pie(
                risk_counts,
                names=risk_col,
                values="Số học sinh",
                hole=0.62,
                title="Cơ cấu mức rủi ro",
                color=risk_col,
                color_discrete_map=risk_colors,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label", marker=dict(line=dict(color="#ffffff", width=3)))
            st.plotly_chart(polish(fig, 430), width="stretch", config=plot_config)

    c1, c2, c3 = st.columns(3)

    with c1:
        if class_col and score_col:
            summary = (
                filtered.groupby(class_col, dropna=False)
                .agg(Students=(score_col, "count"), AvgScore=(score_col, "mean"))
                .reset_index()
                .sort_values("AvgScore", ascending=True)
            )
            fig = px.bar(
                summary,
                x="AvgScore",
                y=class_col,
                orientation="h",
                color="AvgScore",
                text=summary["AvgScore"].map(lambda value: f"{value:.2f}"),
                color_continuous_scale="Tealgrn",
                title="Xếp hạng điểm trung bình theo lớp",
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(range=[0, 10])
            st.plotly_chart(polish(fig, 370), width="stretch", config=plot_config)

    with c2:
        if grade_col and class_col:
            grade_mix = filtered.groupby([class_col, grade_col], dropna=False).size().reset_index(name="Số học sinh")
            fig = px.bar(
                grade_mix,
                x=class_col,
                y="Số học sinh",
                color=grade_col,
                title="Cơ cấu xếp loại theo lớp",
                color_discrete_map=grade_colors,
            )
            st.plotly_chart(polish(fig, 370), width="stretch", config=plot_config)

    with c3:
        if status_col and class_col:
            status_mix = filtered.groupby([class_col, status_col], dropna=False).size().reset_index(name="Số học sinh")
            fig = px.bar(
                status_mix,
                x=class_col,
                y="Số học sinh",
                color=status_col,
                title="Trạng thái theo dõi theo lớp",
                color_discrete_map=status_colors,
            )
            st.plotly_chart(polish(fig, 370), width="stretch", config=plot_config)

    h1, h2 = st.columns([1.2, 1])

    with h1:
        if class_col and test_cols:
            matrix = filtered.groupby(class_col, dropna=False)[test_cols].mean().T
            fig = px.imshow(
                matrix,
                text_auto=".1f",
                aspect="auto",
                zmin=0,
                zmax=10,
                color_continuous_scale="RdYlGn",
                title="Bản đồ nhiệt các đầu điểm",
                labels=dict(x="Lớp", y="Đầu điểm", color="TB"),
            )
            st.plotly_chart(polish(fig, 420), width="stretch", config=plot_config)

    with h2:
        if attendance_col and score_col:
            hover_cols = [col for col in [student_col, class_col, grade_col, status_col, risk_col] if col]
            size_col = violation_col if violation_col else None
            color_col = risk_col if risk_col else class_col
            fig = px.scatter(
                filtered,
                x=attendance_col,
                y=score_col,
                color=color_col,
                size=size_col,
                hover_name=student_col,
                hover_data=hover_cols,
                color_discrete_map=risk_colors if color_col == risk_col else None,
                title="Chuyên cần và điểm trung bình",
                opacity=0.86,
            )
            fig.update_traces(marker=dict(line=dict(width=0.8, color="#ffffff")))
            fig.update_yaxes(range=[0, 10])
            st.plotly_chart(polish(fig, 420), width="stretch", config=plot_config)

with student_tab:
    left, right = st.columns([1.05, 1])

    with left:
        if student_col and progress_col:
            leaders = filtered.nlargest(min(12, len(filtered)), progress_col).sort_values(progress_col, ascending=True)
            fig = px.bar(
                leaders,
                x=progress_col,
                y=student_col,
                orientation="h",
                color=progress_col,
                color_continuous_scale="Viridis",
                title="Học sinh tiến bộ nổi bật",
                hover_data=[col for col in [class_col, score_col, grade_col] if col],
            )
            st.plotly_chart(polish(fig, 460), width="stretch", config=plot_config)

    with right:
        if student_col and score_col:
            rank_cols = [col for col in [student_col, class_col, score_col, attendance_col, progress_col, grade_col, status_col, risk_col] if col]
            rank_table = filtered.sort_values(score_col, ascending=False)[rank_cols].head(12)
            st.markdown('<div class="section-title">Nhóm học sinh nổi bật</div>', unsafe_allow_html=True)
            st.dataframe(rank_table, width="stretch", hide_index=True)

    watch_cols = [col for col in [student_col, class_col, score_col, attendance_col, late_col, violation_col, progress_col, status_col, risk_col, "GhiChu"] if col and col in filtered.columns]
    if watch_cols and (status_col or risk_col):
        watch = filtered.copy()
        watch["_priority"] = 0
        if risk_col:
            watch["_priority"] += watch[risk_col].map({"Cao": 3, "Vua": 2, "Thap": 1}).fillna(0)
        if status_col:
            watch["_priority"] += watch[status_col].map({"Can can thiep": 3, "Theo doi": 2, "On dinh": 1}).fillna(0)
        watch = watch.sort_values(["_priority", score_col if score_col else watch_cols[0]], ascending=[False, True])
        st.markdown('<div class="section-title">Danh sách ưu tiên theo dõi</div>', unsafe_allow_html=True)
        st.dataframe(watch[watch_cols].head(18), width="stretch", hide_index=True)

with analytics_tab:
    a1, a2 = st.columns(2)

    with a1:
        if class_col and score_col:
            fig = px.box(
                filtered,
                x=class_col,
                y=score_col,
                color=class_col,
                points="all",
                title="Độ phân tán điểm theo lớp",
                color_discrete_sequence=palette,
            )
            fig.update_yaxes(range=[0, 10])
            st.plotly_chart(polish(fig, 420), width="stretch", config=plot_config)

    with a2:
        if grade_col and attendance_col:
            fig = px.violin(
                filtered,
                x=grade_col,
                y=attendance_col,
                color=grade_col,
                box=True,
                points="all",
                title="Phân bố chuyên cần theo xếp loại",
                color_discrete_map=grade_colors,
            )
            st.plotly_chart(polish(fig, 420), width="stretch", config=plot_config)

    a3, a4 = st.columns([1.12, 1])

    with a3:
        if len(numeric_cols) >= 2:
            corr_cols = [col for col in numeric_cols if col in filtered.columns]
            corr = filtered[corr_cols].corr(numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Ma trận tương quan các chỉ số",
            )
            st.plotly_chart(polish(fig, 520), width="stretch", config=plot_config)

    with a4:
        if class_col and grade_col:
            treemap_data = filtered.groupby([class_col, grade_col], dropna=False).size().reset_index(name="Số học sinh")
            fig = px.treemap(
                treemap_data,
                path=[class_col, grade_col],
                values="Số học sinh",
                color=grade_col,
                color_discrete_map=grade_colors,
                title="Cấu trúc lớp và xếp loại",
            )
            st.plotly_chart(polish(fig, 520), width="stretch", config=plot_config)

with table_tab:
    st.markdown('<div class="section-title">Dữ liệu đã lọc</div>', unsafe_allow_html=True)
    st.dataframe(filtered, width="stretch", hide_index=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Tải dữ liệu đã lọc",
        csv,
        file_name="student_dashboard_filtered_data.csv",
        mime="text/csv",
        width="content",
    )
