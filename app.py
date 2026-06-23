"""
Disaster Intelligence Dashboard
NLP-Based Disaster Information Extraction & Automated Situation Summarization
"""

import os
import json
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Local imports
from utils.extractor      import process_dataframe
from utils.summarizer     import enrich_dataframe_with_summaries
from utils.powerbi_export import export_for_powerbi, get_powerbi_instructions

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Disaster Intelligence Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Force dark theme across entire app ── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="block-container"],
    .main, .block-container,
    [data-testid="stMain"] {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }
    [data-testid="stSidebar"] .stTextInput input {
        background: #0d1117 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, label,
    [data-testid="stMarkdownContainer"] *,
    .stMarkdown, .stMarkdown * { color: #e6edf3 !important; }
    [data-testid="stTabs"] button { color: #8b949e !important; font-weight: 600; }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }
    [data-testid="stButton"] button {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
    }
    [data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #f85149, #d29922) !important;
        color: #ffffff !important; border: none !important; font-weight: 700 !important;
    }
    [data-testid="stMultiSelect"] > div,
    [data-testid="stSelectbox"] > div {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important; color: #e6edf3 !important;
    }
    [data-testid="stMultiSelect"] span,
    [data-testid="stSelectbox"] span { color: #e6edf3 !important; }
    [data-testid="stExpander"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important; border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary { color: #e6edf3 !important; }
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] * { background-color: #161b22 !important; color: #e6edf3 !important; }
    [data-testid="stAlert"] { background: #161b22 !important; color: #e6edf3 !important; }
    hr { border-color: #30363d !important; }
    [data-testid="stToggle"] label { color: #e6edf3 !important; }
    .metric-card {
        background: #161b22 !important; border: 1px solid #30363d;
        border-radius: 10px; padding: 18px 22px; text-align: center;
    }
    .metric-value { font-size: 2.2rem !important; font-weight: 700 !important; color: #58a6ff !important; }
    .metric-label {
        font-size: 0.8rem !important; color: #8b949e !important;
        text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px;
    }
    .badge-high   { background:#f8514933; color:#f85149 !important; padding:2px 10px;
                    border-radius:20px; font-size:0.78rem; font-weight:600; }
    .badge-medium { background:#d2992233; color:#d29922 !important; padding:2px 10px;
                    border-radius:20px; font-size:0.78rem; font-weight:600; }
    .badge-low    { background:#3fb95033; color:#3fb950 !important; padding:2px 10px;
                    border-radius:20px; font-size:0.78rem; font-weight:600; }
    .summary-box {
        background: #161b22 !important; border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0; padding: 14px 18px; margin-bottom: 10px;
    }
    .summary-box * { color: #e6edf3 !important; }
    .action-box {
        background: #161b22 !important; border-left: 4px solid #d29922;
        border-radius: 0 8px 8px 0; padding: 14px 18px; margin-bottom: 10px;
    }
    .action-box * { color: #e6edf3 !important; }
    .section-header {
        font-size: 1.1rem !important; font-weight: 600 !important;
        color: #e6edf3 !important; border-bottom: 1px solid #30363d;
        padding-bottom: 8px; margin-bottom: 16px;
    }
    [data-testid="stSpinner"] * { color: #58a6ff !important; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚨 Disaster Intelligence")
    st.markdown("**NLP Extraction + AI Summarization**")
    st.divider()

    st.markdown("### ⚙️ Settings")
    api_key = st.text_input(
        "Claude API Key (optional)",
        type="password",
        placeholder="sk-ant-...",
        help="Add your Anthropic API key for AI-powered summaries. Leave empty to use rule-based summaries."
    )

    st.divider()
    st.markdown("### 📂 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload CSV or JSON",
        type=["csv", "json"],
        help="CSV must have a 'text' column. JSON must be a list of objects with 'text' field."
    )

    use_sample = st.button("📋 Load Sample Data", use_container_width=True)

    st.divider()
    st.markdown("### 🔗 Power BI Export")
    export_btn = st.button("⬇️ Export for Power BI", use_container_width=True, disabled=True,
                           key="export_sidebar")
    st.caption("Process data first, then export.")

    st.divider()
    st.markdown(
        "<div style='color:#8b949e;font-size:0.75rem;'>Built with Python · BERT · SpaCy · Streamlit<br>"
        "Academic Project — Jan 2026</div>",
        unsafe_allow_html=True
    )


# ─── State ───────────────────────────────────────────────────────────────────

if "results_df"  not in st.session_state: st.session_state["results_df"]  = None
if "export_path" not in st.session_state: st.session_state["export_path"] = None
if "pbi_guide"   not in st.session_state: st.session_state["pbi_guide"]   = None


# ─── Load Data ───────────────────────────────────────────────────────────────

raw_df = None

if use_sample:
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_disasters.csv")
    raw_df = pd.read_csv(sample_path)
    st.sidebar.success(f"✅ Loaded {len(raw_df)} sample records")

elif uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            data = json.load(uploaded_file)
            raw_df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
        st.sidebar.success(f"✅ Loaded {len(raw_df)} records from {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"❌ Error reading file: {e}")


# ─── Process Button ───────────────────────────────────────────────────────────

if raw_df is not None and "text" not in raw_df.columns:
    # Try to auto-detect text column
    text_cols = [c for c in raw_df.columns if "text" in c.lower() or "content" in c.lower() or "body" in c.lower()]
    if text_cols:
        raw_df = raw_df.rename(columns={text_cols[0]: "text"})
    else:
        st.error("❌ Your file must have a column named **'text'** containing the disaster descriptions.")
        raw_df = None


# ─── Hero Header ─────────────────────────────────────────────────────────────

st.markdown("""
<div style='padding:24px 0 8px 0;'>
    <span style='font-size:2rem;font-weight:800;color:#e6edf3;'>🛰️ Disaster Intelligence Command Centre</span><br>
    <span style='color:#8b949e;font-size:0.95rem;'>
        Real-time NLP extraction · AI situation summarization · Power BI export
    </span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── Main Processing ─────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Records & Summaries", "📡 Live Dashboard", "🧪 Single Text Analysis"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    if raw_df is None:
        st.info("👈 Upload a CSV/JSON file or click **Load Sample Data** from the sidebar to begin.")
        st.markdown("#### Expected CSV Format")
        st.code("id, source, text\nREC-001, Reddit, 'A 6.8 magnitude earthquake struck...'", language="csv")
    else:
        process_col, _ = st.columns([1, 3])
        with process_col:
            process_btn = st.button("⚡ Extract & Analyse", type="primary", use_container_width=True)

        if process_btn or st.session_state["results_df"] is not None:
            if process_btn:
                with st.spinner("🔍 Running NLP extraction pipeline..."):
                    processed = process_dataframe(raw_df, text_col="text")
                    time.sleep(0.3)

                with st.spinner("✍️ Generating AI summaries..."):
                    processed = enrich_dataframe_with_summaries(
                        processed,
                        api_key=api_key if api_key else None
                    )

                st.session_state["results_df"] = processed
                st.success(f"✅ Processed {len(processed)} records successfully!")

            df = st.session_state["results_df"]

            # ── KPI Row ──────────────────────────────────────────────────────
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{len(df)}</div>
                    <div class='metric-label'>Total Incidents</div>
                </div>""", unsafe_allow_html=True)
            with k2:
                high_sev = len(df[df["severity"] == "high"])
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='color:#f85149'>{high_sev}</div>
                    <div class='metric-label'>High Severity</div>
                </div>""", unsafe_allow_html=True)
            with k3:
                high_urg = len(df[df["urgency_level"] == "high"])
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='color:#d29922'>{high_urg}</div>
                    <div class='metric-label'>High Urgency</div>
                </div>""", unsafe_allow_html=True)
            with k4:
                countries = df["country"].nunique()
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='color:#3fb950'>{countries}</div>
                    <div class='metric-label'>Countries Affected</div>
                </div>""", unsafe_allow_html=True)
            with k5:
                avg_conf = df["confidence_score"].mean()
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='color:#58a6ff'>{avg_conf:.0%}</div>
                    <div class='metric-label'>Avg Confidence</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Charts Row 1 ─────────────────────────────────────────────────
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='section-header'>Disaster Types</div>", unsafe_allow_html=True)
                type_counts = df["disaster_type"].value_counts().reset_index()
                type_counts.columns = ["Type", "Count"]
                fig1 = px.bar(
                    type_counts, x="Count", y="Type", orientation="h",
                    color="Count", color_continuous_scale=["#1c2b3a","#58a6ff"],
                    template="plotly_dark",
                )
                fig1.update_layout(
                    plot_bgcolor="#161b22", paper_bgcolor="#161b22",
                    coloraxis_showscale=False, margin=dict(l=0,r=0,t=0,b=0),
                    yaxis=dict(categoryorder="total ascending"),
                    height=280,
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.markdown("<div class='section-header'>Severity Distribution</div>", unsafe_allow_html=True)
                sev_counts = df["severity"].value_counts().reset_index()
                sev_counts.columns = ["Severity", "Count"]
                color_map = {"high": "#f85149", "medium": "#d29922", "low": "#3fb950"}
                fig2 = px.pie(
                    sev_counts, names="Severity", values="Count",
                    color="Severity", color_discrete_map=color_map,
                    template="plotly_dark", hole=0.55,
                )
                fig2.update_layout(
                    plot_bgcolor="#161b22", paper_bgcolor="#161b22",
                    margin=dict(l=0,r=0,t=0,b=0), height=280,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                )
                st.plotly_chart(fig2, use_container_width=True)

            # ── Charts Row 2 ─────────────────────────────────────────────────
            col3, col4 = st.columns(2)

            with col3:
                st.markdown("<div class='section-header'>Urgency Level Breakdown</div>", unsafe_allow_html=True)
                urg_counts = df["urgency_level"].value_counts().reset_index()
                urg_counts.columns = ["Urgency", "Count"]
                urg_color = {"high": "#f85149", "medium": "#d29922", "low": "#3fb950"}
                fig3 = px.bar(
                    urg_counts, x="Urgency", y="Count",
                    color="Urgency", color_discrete_map=urg_color,
                    template="plotly_dark",
                )
                fig3.update_layout(
                    plot_bgcolor="#161b22", paper_bgcolor="#161b22",
                    showlegend=False, margin=dict(l=0,r=0,t=0,b=0), height=260,
                )
                st.plotly_chart(fig3, use_container_width=True)

            with col4:
                st.markdown("<div class='section-header'>Countries Affected</div>", unsafe_allow_html=True)
                country_df = df[df["country"] != "unknown"]["country"].value_counts().reset_index()
                country_df.columns = ["Country", "Incidents"]
                if not country_df.empty:
                    fig4 = px.bar(
                        country_df, x="Incidents", y="Country", orientation="h",
                        color="Incidents", color_continuous_scale=["#1c3a2b","#3fb950"],
                        template="plotly_dark",
                    )
                    fig4.update_layout(
                        plot_bgcolor="#161b22", paper_bgcolor="#161b22",
                        coloraxis_showscale=False, margin=dict(l=0,r=0,t=0,b=0),
                        yaxis=dict(categoryorder="total ascending"), height=260,
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("No country data extracted.")

            # ── Resources Needed ─────────────────────────────────────────────
            st.markdown("<div class='section-header'>Resources Needed Across All Incidents</div>", unsafe_allow_html=True)
            all_resources = []
            for r in df["resources_str"].dropna():
                all_resources.extend([x.strip() for x in r.split(",") if x.strip() != "unknown"])
            if all_resources:
                res_series = pd.Series(all_resources).value_counts().reset_index()
                res_series.columns = ["Resource", "Count"]
                fig5 = px.treemap(
                    res_series, path=["Resource"], values="Count",
                    color="Count", color_continuous_scale=["#1c2b3a","#58a6ff"],
                    template="plotly_dark",
                )
                fig5.update_layout(
                    paper_bgcolor="#161b22", margin=dict(l=0,r=0,t=0,b=0), height=250,
                )
                st.plotly_chart(fig5, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — RECORDS & SUMMARIES
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    df = st.session_state.get("results_df")
    if df is None:
        st.info("Process your data in the Dashboard tab first.")
    else:
        st.markdown(f"### 📋 {len(df)} Processed Records")

        # Filter controls
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sev_filter = st.multiselect("Filter by Severity", ["high","medium","low"], default=["high","medium","low"])
        with fc2:
            urg_filter = st.multiselect("Filter by Urgency", ["high","medium","low"], default=["high","medium","low"])
        with fc3:
            type_filter = st.multiselect("Filter by Disaster Type", df["disaster_type"].unique().tolist(),
                                          default=df["disaster_type"].unique().tolist())

        filtered = df[
            df["severity"].isin(sev_filter) &
            df["urgency_level"].isin(urg_filter) &
            df["disaster_type"].isin(type_filter)
        ]

        st.markdown(f"*Showing {len(filtered)} records*")
        st.divider()

        # Card view
        for _, row in filtered.iterrows():
            badge_sev = f"<span class='badge-{row['severity']}'>{row['severity'].upper()}</span>"
            badge_urg = f"<span class='badge-{row['urgency_level']}'>{row['urgency_level'].upper()} URGENCY</span>"
            with st.expander(
                f"🔴 {row['record_id']} — {row['disaster_type'].upper()} | {row['city']}, {row['country']}",
                expanded=False
            ):
                m1, m2, m3 = st.columns(3)
                m1.markdown(f"**Disaster Type:** {row['disaster_type'].title()}")
                m2.markdown(f"**Severity:** {badge_sev}", unsafe_allow_html=True)
                m3.markdown(f"**Urgency:** {badge_urg}", unsafe_allow_html=True)

                m4, m5, m6 = st.columns(3)
                m4.markdown(f"**Location:** {row['city']}, {row['country']}")
                m5.markdown(f"**People Affected:** {row['people_affected']}")
                m6.markdown(f"**Confidence:** {row['confidence_score']:.0%}")

                st.markdown(f"**Resources Needed:** `{row['resources_str']}`")
                st.markdown(f"**Organizations:** `{row['organizations']}`")

                st.markdown(f"""<div class='summary-box'>
                    🗒️ <strong>Situation Summary</strong><br>{row.get('summary', 'Not available')}
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""<div class='action-box'>
                    ⚡ <strong>Recommended Action</strong><br>{row.get('recommended_action', 'Not available')}
                </div>""", unsafe_allow_html=True)

                st.caption(f"Raw text: {row.get('raw_text','')[:200]}...")

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — POWER BI EXPORT
# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — LIVE DASHBOARD (Power BI–style, auto-refreshing)
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    df = st.session_state.get("results_df")
    if df is None:
        st.info("⚡ Process your data in the **Dashboard** tab first, then come back here for the live view.")
    else:
        # ── Live Dashboard Header ────────────────────────────────────────────
        head_col, ctrl_col = st.columns([3, 1])
        with head_col:
            st.markdown("""
            <div style='padding:6px 0 2px 0;'>
                <span style='font-size:1.4rem;font-weight:700;color:#e6edf3;'>
                    📡 Live Situation Dashboard
                </span>
                <span style='font-size:0.8rem;color:#3fb950;margin-left:12px;'>● LIVE</span>
            </div>
            <div style='color:#8b949e;font-size:0.82rem;margin-bottom:4px;'>
                Auto-refreshes every 30 seconds · Power BI–style analytics
            </div>
            """, unsafe_allow_html=True)

        with ctrl_col:
            refresh_rate = st.selectbox("Refresh every", [10, 30, 60, 120],
                                         index=1, format_func=lambda x: f"{x}s")
            auto_refresh = st.toggle("Auto-refresh", value=True)

        # Export CSV button (slim, top-right area)
        export_path = st.session_state.get("export_path")
        dl_col1, dl_col2, _ = st.columns([1, 1, 3])
        with dl_col1:
            if st.button("⬇️ Export CSV", use_container_width=True):
                path = export_for_powerbi(df, filename="disaster_latest.csv")
                st.session_state["export_path"] = path
                st.success("✅ Exported!")
        with dl_col2:
            ep = st.session_state.get("export_path")
            if ep and os.path.exists(ep):
                with open(ep, "rb") as f:
                    st.download_button("📥 Download", data=f,
                                       file_name="disaster_latest.csv",
                                       mime="text/csv",
                                       use_container_width=True)

        st.divider()

        # ── Slicer row (Power BI–style filters) ─────────────────────────────
        st.markdown("<div style='color:#8b949e;font-size:0.78rem;text-transform:uppercase;"
                    "letter-spacing:0.1em;margin-bottom:6px;'>FILTERS</div>",
                    unsafe_allow_html=True)
        sl1, sl2, sl3 = st.columns(3)
        with sl1:
            sel_type = st.multiselect("Disaster Type", df["disaster_type"].unique().tolist(),
                                       default=df["disaster_type"].unique().tolist(), key="live_type")
        with sl2:
            sel_sev  = st.multiselect("Severity", ["high","medium","low"],
                                       default=["high","medium","low"], key="live_sev")
        with sl3:
            sel_urg  = st.multiselect("Urgency", ["high","medium","low"],
                                       default=["high","medium","low"], key="live_urg")

        live_df = df[
            df["disaster_type"].isin(sel_type) &
            df["severity"].isin(sel_sev) &
            df["urgency_level"].isin(sel_urg)
        ]

        st.markdown(f"<div style='color:#8b949e;font-size:0.8rem;margin-bottom:8px;'>"
                    f"Showing <strong style='color:#e6edf3;'>{len(live_df)}</strong> of "
                    f"{len(df)} incidents</div>", unsafe_allow_html=True)

        # ── KPI Cards ───────────────────────────────────────────────────────
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        total       = len(live_df)
        high_sev    = len(live_df[live_df["severity"]     == "high"])
        high_urg    = len(live_df[live_df["urgency_level"] == "high"])
        countries   = live_df["country"].nunique()
        known_ppl   = pd.to_numeric(live_df["people_affected"], errors="coerce").dropna()
        total_ppl   = int(known_ppl.sum()) if len(known_ppl) > 0 else 0
        avg_conf    = live_df["confidence_score"].mean()

        for col, val, label, color in [
            (k1, total,       "Incidents",       "#58a6ff"),
            (k2, high_sev,    "High Severity",   "#f85149"),
            (k3, high_urg,    "High Urgency",    "#d29922"),
            (k4, countries,   "Countries",       "#3fb950"),
            (k5, f"{total_ppl:,}" if total_ppl else "N/A", "Est. Affected", "#a371f7"),
            (k6, f"{avg_conf:.0%}", "Avg Confidence", "#58a6ff"),
        ]:
            col.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='color:{color};font-size:1.8rem;'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Gauge + Donut + Bar ───────────────────────────────────────
        g1, g2, g3 = st.columns([1, 1, 1])

        with g1:
            # Urgency Gauge — Power BI–style
            urg_score = round((high_urg / max(total, 1)) * 100)
            gauge_color = "#f85149" if urg_score > 60 else "#d29922" if urg_score > 30 else "#3fb950"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=urg_score,
                title={"text": "High Urgency %", "font": {"color": "#e6edf3", "size": 13}},
                delta={"reference": 50, "increasing": {"color": "#f85149"},
                       "decreasing": {"color": "#3fb950"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b949e",
                             "tickfont": {"color": "#8b949e"}},
                    "bar": {"color": gauge_color},
                    "bgcolor": "#161b22",
                    "bordercolor": "#30363d",
                    "steps": [
                        {"range": [0,   30],  "color": "#1a2f1a"},
                        {"range": [30,  60],  "color": "#2f2a14"},
                        {"range": [60,  100], "color": "#2f1414"},
                    ],
                    "threshold": {"line": {"color": "#ffffff", "width": 2},
                                  "thickness": 0.75, "value": 70},
                },
                number={"suffix": "%", "font": {"color": "#e6edf3", "size": 28}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#161b22", font_color="#e6edf3",
                margin=dict(l=20, r=20, t=40, b=10), height=240,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g2:
            # Severity donut
            sev_counts = live_df["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity","Count"]
            fig_donut = px.pie(
                sev_counts, names="Severity", values="Count", hole=0.6,
                color="Severity",
                color_discrete_map={"high":"#f85149","medium":"#d29922","low":"#3fb950"},
                template="plotly_dark",
                title="Severity Split",
            )
            fig_donut.update_traces(textfont_color="#e6edf3")
            fig_donut.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                margin=dict(l=0,r=0,t=40,b=0), height=240,
                title_font_color="#e6edf3", title_font_size=13,
                legend=dict(orientation="h", y=-0.1, font_color="#8b949e"),
                annotations=[dict(text=f"<b>{total}</b><br>total", x=0.5, y=0.5,
                                  font_size=14, font_color="#e6edf3", showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with g3:
            # Disaster type horizontal bar
            type_counts = live_df["disaster_type"].value_counts().reset_index()
            type_counts.columns = ["Type","Count"]
            fig_bar = px.bar(
                type_counts, x="Count", y="Type", orientation="h",
                color="Count", color_continuous_scale=["#1c2b3a","#58a6ff"],
                template="plotly_dark", title="Incident Types",
            )
            fig_bar.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                coloraxis_showscale=False,
                margin=dict(l=0,r=0,t=40,b=0), height=240,
                yaxis=dict(categoryorder="total ascending",
                           tickfont_color="#8b949e"),
                xaxis=dict(tickfont_color="#8b949e"),
                title_font_color="#e6edf3", title_font_size=13,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Row 2: Stacked bar + Scatter ─────────────────────────────────────
        r2a, r2b = st.columns([3, 2])

        with r2a:
            # Stacked bar — disaster type × urgency (Power BI matrix style)
            stack_df = live_df.groupby(["disaster_type","urgency_level"]).size().reset_index(name="Count")
            fig_stack = px.bar(
                stack_df, x="disaster_type", y="Count", color="urgency_level",
                color_discrete_map={"high":"#f85149","medium":"#d29922","low":"#3fb950"},
                template="plotly_dark", barmode="stack",
                title="Incidents by Type × Urgency",
            )
            fig_stack.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                margin=dict(l=0,r=0,t=40,b=0), height=280,
                title_font_color="#e6edf3", title_font_size=13,
                legend=dict(title="Urgency", font_color="#8b949e"),
                xaxis=dict(tickfont_color="#8b949e", tickangle=-20),
                yaxis=dict(tickfont_color="#8b949e"),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

        with r2b:
            # Scatter — confidence vs severity_score
            scatter_df = live_df.copy()
            scatter_df["sev_num"] = scatter_df["severity"].map({"low":1,"medium":2,"high":3})
            fig_scatter = px.scatter(
                scatter_df, x="sev_num", y="confidence_score",
                color="disaster_type", size_max=14,
                template="plotly_dark",
                title="Confidence vs Severity",
                labels={"sev_num":"Severity (1=Low, 3=High)",
                        "confidence_score":"Confidence Score"},
                hover_data=["city","country","urgency_level"],
            )
            fig_scatter.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                margin=dict(l=0,r=0,t=40,b=0), height=280,
                title_font_color="#e6edf3", title_font_size=13,
                xaxis=dict(tickvals=[1,2,3], ticktext=["Low","Medium","High"],
                           tickfont_color="#8b949e"),
                yaxis=dict(tickfont_color="#8b949e"),
                legend=dict(font_color="#8b949e"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Row 3: Resources treemap + Country bar ───────────────────────────
        r3a, r3b = st.columns([2, 2])

        with r3a:
            all_res = []
            for r in live_df["resources_str"].dropna():
                all_res.extend([x.strip() for x in r.split(",") if x.strip() not in ("unknown","")])
            if all_res:
                res_df = pd.Series(all_res).value_counts().reset_index()
                res_df.columns = ["Resource","Count"]
                fig_tree = px.treemap(
                    res_df, path=["Resource"], values="Count",
                    color="Count", color_continuous_scale=["#1c2b3a","#a371f7"],
                    template="plotly_dark", title="Resources Needed",
                )
                fig_tree.update_layout(
                    paper_bgcolor="#161b22",
                    margin=dict(l=0,r=0,t=40,b=0), height=260,
                    title_font_color="#e6edf3", title_font_size=13,
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_tree, use_container_width=True)

        with r3b:
            country_df = live_df[live_df["country"] != "unknown"]["country"].value_counts().reset_index()
            country_df.columns = ["Country","Incidents"]
            if not country_df.empty:
                fig_country = px.bar(
                    country_df, x="Country", y="Incidents",
                    color="Incidents",
                    color_continuous_scale=["#1c3a2b","#3fb950"],
                    template="plotly_dark", title="Incidents by Country",
                )
                fig_country.update_layout(
                    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                    coloraxis_showscale=False,
                    margin=dict(l=0,r=0,t=40,b=0), height=260,
                    title_font_color="#e6edf3", title_font_size=13,
                    xaxis=dict(tickfont_color="#8b949e"),
                    yaxis=dict(tickfont_color="#8b949e"),
                )
                st.plotly_chart(fig_country, use_container_width=True)

        # ── Live Data Table (Power BI table visual style) ────────────────────
        st.markdown("<div class='section-header' style='margin-top:8px;'>🗃️ Live Records Table</div>",
                    unsafe_allow_html=True)

        table_df = live_df[["record_id","disaster_type","city","country",
                             "severity","urgency_level","people_affected",
                             "resources_str","confidence_score","summary"]].copy()
        table_df.columns = ["ID","Type","City","Country","Severity",
                            "Urgency","Affected","Resources","Confidence","Summary"]
        table_df["Confidence"] = table_df["Confidence"].apply(lambda x: f"{x:.0%}")

        st.dataframe(
            table_df,
            use_container_width=True,
            height=280,
            column_config={
                "Severity":   st.column_config.TextColumn("Severity"),
                "Urgency":    st.column_config.TextColumn("Urgency"),
                "Confidence": st.column_config.TextColumn("Confidence"),
                "Summary":    st.column_config.TextColumn("Summary", width="large"),
            }
        )

        # ── Timestamp + Auto-refresh ─────────────────────────────────────────
        st.markdown(
            f"<div style='color:#3fb950;font-size:0.78rem;margin-top:8px;'>"
            f"● Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')} · "
            f"Next refresh in {refresh_rate}s</div>",
            unsafe_allow_html=True
        )

        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — SINGLE TEXT ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🧪 Analyse a Single Disaster Report")
    st.caption("Paste any unstructured text to extract disaster information instantly.")

    example_texts = {
        "Earthquake — Turkey":     "A 6.8 magnitude earthquake struck coastal Turkey early this morning, collapsing several buildings in Izmir. Rescue teams are searching for survivors and at least 500 families have been displaced.",
        "Flood — Bangladesh":      "Catastrophic flooding has submerged large parts of Bangladesh after heavy monsoon rains. Over 200,000 people have been displaced and drinking water is critically scarce.",
        "Wildfire — Los Angeles":  "Massive wildfire burning near Los Angeles has forced evacuation of 15,000 residents. Fire departments are battling the blaze.",
    }
    chosen = st.selectbox("Load an example or type your own below:", ["Custom"] + list(example_texts.keys()))
    default_text = example_texts.get(chosen, "")

    input_text = st.text_area("Disaster Report Text:", value=default_text, height=130,
                               placeholder="Paste unstructured disaster text here...")

    if st.button("🔍 Analyse Text", type="primary"):
        if not input_text.strip():
            st.warning("Please enter some text first.")
        else:
            from utils.extractor  import extract_disaster_info
            from utils.summarizer import summarize_with_claude, fallback_summary

            with st.spinner("Extracting..."):
                info = extract_disaster_info(input_text, source="manual")

            with st.spinner("Summarizing..."):
                if api_key:
                    ai_result = summarize_with_claude({
                        **info,
                        "city":          info["location"].get("city","unknown"),
                        "country":       info["location"].get("country","unknown"),
                        "resources_str": ", ".join(info["resources_needed"]),
                    }, api_key)
                else:
                    ai_result = fallback_summary({
                        **info,
                        "city":          info["location"].get("city","unknown"),
                        "country":       info["location"].get("country","unknown"),
                        "resources_str": ", ".join(info["resources_needed"]),
                    })

            st.divider()

            r1, r2, r3 = st.columns(3)
            r1.metric("Disaster Type",   info["disaster_type"].title())
            r2.metric("Severity",        info["severity"].upper())
            r3.metric("Urgency",         info["urgency_level"].upper())

            r4, r5, r6 = st.columns(3)
            r4.metric("City",            info["location"].get("city","unknown").title())
            r5.metric("Country",         info["location"].get("country","unknown").title())
            r6.metric("Confidence",      f"{info['confidence_score']:.0%}")

            st.markdown(f"**People Affected:** `{info['people_affected']}`")
            st.markdown(f"**Resources Needed:** `{', '.join(info['resources_needed'])}`")

            st.markdown(f"""<div class='summary-box'>
                🗒️ <strong>Situation Summary</strong><br>{ai_result.get('summary','')}
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class='action-box'>
                ⚡ <strong>Recommended Action</strong><br>{ai_result.get('recommended_action','')}
            </div>""", unsafe_allow_html=True)

            with st.expander("📄 Raw JSON Output"):
                info["summary"]            = ai_result.get("summary","")
                info["recommended_action"] = ai_result.get("recommended_action","")
                st.json(info)
