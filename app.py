"""
app.py — Social Media Engagement Intelligence Dashboard
=======================================================
Run with:  streamlit run app.py

Tabs:
  1. Overview        — KPI cards + engagement timeline
  2. Platform Intel  — Platform-level comparison charts
  3. Content Engine  — Content type + category performance
  4. Best Time       — Heatmap + hour/day analysis
  5. Trending Posts  — Viral post detection
  6. ML Predictions  — Predict vs actual + feature importance
  7. Anomaly Radar   — Multi-signal anomaly detection
  8. AI Insights     — Auto-generated insight cards
"""

import os
import sys
import io
import warnings
import zipfile

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Project imports ──────────────────────────────────────────────────────────
from data.data_generator      import load_or_generate, generate_dataset
from modules.analytics        import (
    compute_kpis, engagement_over_time, engagement_by_platform_over_time,
    platform_comparison, content_type_performance, best_time_heatmap,
    best_posting_hours, detect_trending_posts, hashtag_impact,
    category_performance, follower_tier_analysis
)
from modules.ml_models        import EngagementPredictor, AnomalyDetector, engineer_features
from modules.insights         import generate_all_insights

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SocialIQ — Engagement Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── LOAD CSS ─────────────────────────────────────────────────────────────────

def load_css(path: str):
    if os.path.exists(path):
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")

# ─── PLOTLY THEME ────────────────────────────────────────────────────────────

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PALETTE   = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b",
                   "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6"]
CHART_BG        = "rgba(17,24,39,0.95)"

def styled_fig(fig, title: str = "", height: int = 380):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor =CHART_BG,
        font=dict(family="Space Grotesk, sans-serif", color="#f1f5f9"),
        title=dict(text=title, font=dict(size=14, color="#f1f5f9")),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        height=height,
    )
    return fig


# ─── DATA LOADING & CACHING ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_data(n_posts=2000) -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)
    df = load_or_generate("data/social_media_data.csv", n_posts=n_posts)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_resource(show_spinner=False)
def get_ml_pipeline(df: pd.DataFrame):
    """Train ML pipeline (cached so it runs only once per session)."""
    feature_df, encoders = engineer_features(df)
    predictor = EngagementPredictor()
    predictor.fit(feature_df)
    detector  = AnomalyDetector(contamination=0.03)
    df_anomaly = detector.fit_transform(df)
    return predictor, detector, df_anomaly, feature_df


# ─── HELPER: HTML COMPONENTS ─────────────────────────────────────────────────

def metric_card(label, value, delta="", icon="📊"):
    return f"""
    <div class="metric-card">
        <div class="icon">{icon}</div>
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {'<div class="delta">' + delta + '</div>' if delta else ''}
    </div>"""

def section_header(title, badge=""):
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    return f"""<div class="section-header">
        <h2>{title}</h2>{badge_html}
    </div>"""

def trending_card(row):
    return f"""
    <div class="trending-card">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.3rem">
            <span class="post-id">{row['post_id']}</span>
            <span class="platform-badge">{row['platform']}</span>
        </div>
        <div class="engagement-num">
            {int(row['total_engagement']):,} engagements
            <span style="font-size:0.8rem;color:#64748b;font-weight:400">
              · {row['content_type']} · {row['category']}
            </span>
        </div>
        <div class="explanation">💡 {row['explanation']}</div>
        <div style="font-size:0.72rem;color:#64748b;margin-top:0.4rem">
            {row['timestamp'].strftime('%d %b %Y, %H:%M')} &nbsp;|&nbsp;
            ER: <b style="color:#6366f1">{row['engagement_rate']:.2f}%</b> &nbsp;|&nbsp;
            Z-score: <b style="color:#f59e0b">{row.get('z_score', 0):.2f}σ</b>
        </div>
    </div>"""

def insight_card(ins):
    return f"""
    <div class="insight-card">
        <span class="insight-icon">{ins['icon']}</span>
        <div class="insight-body">
            <div class="headline">{ins['headline']}</div>
            <div class="detail">{ins['detail']}</div>
            <span class="cat-badge">{ins['category']}</span>
        </div>
    </div>"""


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

with st.spinner("🔄 Loading social media intelligence engine…"):
    raw_df = get_data(n_posts=2000)
    predictor, detector, df_anomaly, feature_df = get_ml_pipeline(raw_df)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1.5rem">
        <div style="font-size:2rem">📊</div>
        <div style="font-size:1.1rem;font-weight:700;
                    background:linear-gradient(135deg,#6366f1,#06b6d4);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            SocialIQ
        </div>
        <div style="font-size:0.72rem;color:#64748b;margin-top:0.1rem">
            Engagement Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🔍 FILTERS**")

    # Platform
    platforms = sorted(raw_df["platform"].unique())
    sel_platforms = st.multiselect(
        "Platform", platforms, default=platforms,
        help="Select one or more platforms"
    )

    # Date range
    min_date = raw_df["timestamp"].min().date()
    max_date = raw_df["timestamp"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
    )

    # Content type
    content_types = sorted(raw_df["content_type"].unique())
    sel_content = st.multiselect(
        "Content Type", content_types, default=content_types
    )

    # Category
    categories = sorted(raw_df["category"].unique())
    sel_categories = st.multiselect(
        "Category", categories, default=categories
    )

    st.markdown("---")

    # Regenerate button
    if st.button("🔄 Regenerate Dataset"):
        st.cache_data.clear()
        st.cache_resource.clear()
        if os.path.exists("data/social_media_data.csv"):
            os.remove("data/social_media_data.csv")
        st.rerun()

    st.markdown("---")
    st.markdown("""<div style="font-size:0.7rem;color:#475569;text-align:center">
        Built with Python · Streamlit · scikit-learn<br>
        Portfolio-grade project
    </div>""", unsafe_allow_html=True)


# ─── APPLY FILTERS ───────────────────────────────────────────────────────────

def apply_filters(df):
    mask = pd.Series([True] * len(df), index=df.index)
    if sel_platforms:
        mask &= df["platform"].isin(sel_platforms)
    if len(date_range) == 2:
        start, end = date_range
        mask &= df["timestamp"].dt.date.between(start, end)
    if sel_content:
        mask &= df["content_type"].isin(sel_content)
    if sel_categories:
        mask &= df["category"].isin(sel_categories)
    return df[mask].copy()

df = apply_filters(df_anomaly)

if df.empty:
    st.warning("⚠️ No data matches your current filters. Please adjust the sidebar filters.")
    st.stop()


# ─── PAGE HEADER ─────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom:1.5rem">
    <div class="page-title">📊 Social Media Engagement Intelligence</div>
    <div class="page-subtitle">
        Real-time analytics · ML predictions · Anomaly detection · AI insights
    </div>
</div>
""", unsafe_allow_html=True)

kpis = compute_kpis(df)

# ─── KPI CARDS ROW ───────────────────────────────────────────────────────────

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.markdown(metric_card("Total Posts", f"{kpis['total_posts']:,}", icon="📝"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Total Likes", f"{kpis['total_likes']:,}", icon="❤️"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Total Comments", f"{kpis['total_comments']:,}", icon="💬"), unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("Total Shares", f"{kpis['total_shares']:,}", icon="🔁"), unsafe_allow_html=True)
with c5:
    st.markdown(metric_card("Avg Eng. Rate", f"{kpis['avg_eng_rate']}%", icon="📈"), unsafe_allow_html=True)
with c6:
    st.markdown(metric_card("Best Platform", kpis['best_platform'], icon="🏆"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── TABS ─────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📈 Overview",
    "🌐 Platform Intel",
    "🎨 Content Engine",
    "⏰ Best Time",
    "🔥 Trending Posts",
    "🤖 ML Predictions",
    "⚠️ Anomaly Radar",
    "💡 AI Insights",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(section_header("Engagement Over Time", "DAILY"), unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        ts_data = engagement_over_time(df, freq="D")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_data["date"], y=ts_data["engagement"],
            name="Daily Engagement",
            fill="tozeroy", fillcolor="rgba(99,102,241,0.12)",
            line=dict(color="#6366f1", width=2),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=ts_data["date"], y=ts_data["rolling_avg"],
            name="7-day Rolling Avg",
            line=dict(color="#06b6d4", width=2, dash="dot"),
            mode="lines",
        ))
        styled_fig(fig, "Total Engagement Over Time", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Engagement distribution
        fig2 = go.Figure(go.Box(
            y=df["engagement_rate"],
            name="Engagement Rate",
            marker_color="#6366f1",
            boxmean="sd",
        ))
        styled_fig(fig2, "Engagement Rate Distribution", height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Platform time series
    st.markdown(section_header("Platform Engagement Over Time", "WEEKLY"), unsafe_allow_html=True)
    pt_ts = engagement_by_platform_over_time(df, freq="W")
    fig3 = px.line(
        pt_ts, x="date", y="total_engagement", color="platform",
        color_discrete_sequence=COLOR_PALETTE,
    )
    fig3.update_traces(line=dict(width=2))
    styled_fig(fig3, "", height=320)
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PLATFORM INTEL
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    pc = platform_comparison(df)

    st.markdown(section_header("Platform KPI Comparison", "OVERVIEW"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            pc.sort_values("avg_eng_rate", ascending=True),
            x="avg_eng_rate", y="platform", orientation="h",
            color="avg_eng_rate", color_continuous_scale="Viridis",
            text="avg_eng_rate",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_coloraxes(showscale=False)
        styled_fig(fig, "Average Engagement Rate by Platform", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            pc, x="platform",
            y=["avg_likes", "avg_comments", "avg_shares"],
            barmode="group",
            color_discrete_sequence=COLOR_PALETTE,
        )
        styled_fig(fig2, "Avg Likes / Comments / Shares by Platform", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # Radar chart
    st.markdown(section_header("Platform Radar — Multi-Metric"), unsafe_allow_html=True)
    metrics = ["avg_likes", "avg_comments", "avg_shares", "avg_eng_rate"]
    norm_pc = pc.copy()
    for m in metrics:
        norm_pc[m] = (pc[m] - pc[m].min()) / (pc[m].max() - pc[m].min() + 1e-9)

    fig_radar = go.Figure()
    for i, row in norm_pc.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics] + [row[metrics[0]]],
            theta=["Likes", "Comments", "Shares", "Eng Rate", "Likes"],
            name=row["platform"],
            line_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            fill="toself", fillcolor=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            opacity=0.25,
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(17,24,39,0.9)",
            radialaxis=dict(visible=True, range=[0, 1],
                            color="#475569", gridcolor="#1e293b"),
            angularaxis=dict(color="#94a3b8", gridcolor="#1e293b"),
        ),
        template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG,
        font=dict(color="#f1f5f9"), height=380,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONTENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    ct = content_type_performance(df)
    cat = category_performance(df)
    ht  = hashtag_impact(df, top_n=15)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(section_header("Content Type Performance"), unsafe_allow_html=True)
        fig = px.pie(
            ct, values="total_eng", names="content_type",
            color_discrete_sequence=COLOR_PALETTE, hole=0.45,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        styled_fig(fig, "Content Type Share of Total Engagement", height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(section_header("Avg Engagement Rate by Type"), unsafe_allow_html=True)
        ct_sorted = ct.sort_values("avg_eng_rate", ascending=False)
        fig2 = px.bar(
            ct_sorted, x="content_type", y="avg_eng_rate",
            color="content_type", color_discrete_sequence=COLOR_PALETTE,
            text="avg_eng_rate",
        )
        fig2.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig2.update_layout(showlegend=False)
        styled_fig(fig2, "", height=340)
        st.plotly_chart(fig2, use_container_width=True)

    # Category heatmap-style bar
    st.markdown(section_header("Category Performance", "ALL CATEGORIES"), unsafe_allow_html=True)
    fig3 = px.bar(
        cat.sort_values("avg_eng_rate", ascending=True),
        x="avg_eng_rate", y="category", orientation="h",
        color="avg_eng_rate", color_continuous_scale="Plasma",
        text="avg_eng_rate",
    )
    fig3.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig3.update_coloraxes(showscale=False)
    styled_fig(fig3, "Average Engagement Rate by Category", height=380)
    st.plotly_chart(fig3, use_container_width=True)

    # Hashtag impact
    st.markdown(section_header("Hashtag Impact Analysis", "TOP 15"), unsafe_allow_html=True)
    fig4 = px.bar(
        ht.sort_values("avg_eng_rate"),
        x="avg_eng_rate", y="hashtag", orientation="h",
        color="avg_eng_rate", color_continuous_scale="Cividis",
        hover_data=["posts", "avg_likes", "avg_shares"],
    )
    fig4.update_coloraxes(showscale=False)
    styled_fig(fig4, "Top Hashtags by Average Engagement Rate", height=480)
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BEST TIME TO POST
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(section_header("Best Time to Post — Heatmap", "HOUR × DAY"), unsafe_allow_html=True)

    heatmap_data = best_time_heatmap(df)
    fig = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns.tolist(),
        y=[f"{h:02d}:00" for h in heatmap_data.index],
        colorscale="Plasma",
        colorbar=dict(title="Avg ER %", tickfont=dict(color="#94a3b8")),
        text=[[f"{v:.2f}%" for v in row] for row in heatmap_data.values],
        texttemplate="%{text}",
        textfont=dict(size=9),
        hoverongaps=False,
    ))
    styled_fig(fig, "Engagement Rate by Hour and Day of Week", height=550)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(section_header("Top 5 Hours"), unsafe_allow_html=True)
        best_h = best_posting_hours(df, top_n=8)
        fig2 = px.bar(
            best_h, x="hour", y="engagement_rate",
            color="engagement_rate", color_continuous_scale="Viridis",
            text="engagement_rate",
        )
        fig2.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig2.update_coloraxes(showscale=False)
        fig2.update_xaxes(tickvals=best_h["hour"],
                          ticktext=[f"{h}:00" for h in best_h["hour"]])
        styled_fig(fig2, "Best Hours to Post", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown(section_header("Day of Week Analysis"), unsafe_allow_html=True)
        day_data = (
            df.groupby("weekday")["engagement_rate"].mean()
              .reindex(["Monday","Tuesday","Wednesday","Thursday",
                        "Friday","Saturday","Sunday"])
              .reset_index()
        )
        fig3 = px.bar(
            day_data, x="weekday", y="engagement_rate",
            color="engagement_rate", color_continuous_scale="Turbo",
        )
        fig3.update_coloraxes(showscale=False)
        styled_fig(fig3, "Avg Engagement Rate by Day", height=300)
        st.plotly_chart(fig3, use_container_width=True)

    # Insight callout
    peak_hour = best_h.iloc[0]["hour"]
    peak_er   = best_h.iloc[0]["engagement_rate"]
    st.info(
        f"⏰ **Optimal Posting Window:** Posts published at **{int(peak_hour)}:00** "
        f"average **{peak_er:.2f}%** engagement rate — "
        f"the highest in your dataset. Schedule content for this window to maximize reach."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRENDING POSTS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown(section_header("Trending Posts Detection", "IQR SPIKE DETECTION"),
                unsafe_allow_html=True)

    multiplier = st.slider(
        "Spike sensitivity (lower = more posts flagged)",
        min_value=1.5, max_value=5.0, value=2.5, step=0.25
    )
    trending = detect_trending_posts(df, multiplier=multiplier, top_n=12)

    if trending.empty:
        st.warning("No trending posts found with current sensitivity. Try lowering the slider.")
    else:
        # Summary bar
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(metric_card("Trending Posts Found",
                                    str(len(trending)), icon="🔥"), unsafe_allow_html=True)
        with col2:
            top_platform = trending["platform"].mode()[0]
            st.markdown(metric_card("Top Trending Platform",
                                    top_platform, icon="🌐"), unsafe_allow_html=True)
        with col3:
            max_eng = int(trending["total_engagement"].max())
            st.markdown(metric_card("Peak Engagement",
                                    f"{max_eng:,}", icon="⚡"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Trending posts chart
        fig = px.scatter(
            trending,
            x="timestamp", y="total_engagement",
            size="engagement_rate", color="platform",
            hover_data=["post_id", "content_type", "hashtags", "explanation"],
            color_discrete_sequence=COLOR_PALETTE,
            size_max=40,
        )
        styled_fig(fig, "Trending Post Scatter — Size = Engagement Rate", height=340)
        st.plotly_chart(fig, use_container_width=True)

        # Cards
        st.markdown(section_header("Trending Post Cards"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for i, (_, row) in enumerate(trending.iterrows()):
            with (c1 if i % 2 == 0 else c2):
                st.markdown(trending_card(row), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ML PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown(section_header("ML Prediction Engine", "RIDGE + RANDOM FOREST"),
                unsafe_allow_html=True)

    target_sel = st.selectbox(
        "Select target variable to predict:",
        ["engagement_rate", "likes", "comments"],
        format_func=lambda x: x.replace("_", " ").title()
    )
    model_sel = st.radio("Model", ["rf", "ridge"],
                         format_func=lambda x: "Random Forest" if x=="rf" else "Ridge Regression",
                         horizontal=True)

    metrics = predictor.metrics[target_sel][model_sel]

    # Metric callouts
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(metric_card("MAE", str(metrics["mae"]), icon="📉"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("RMSE", str(metrics["rmse"]), icon="📊"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("R² Score", str(metrics["r2"]), icon="🎯"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        # Predicted vs Actual scatter
        pva = predictor.predicted_vs_actual(target_sel, model_type=model_sel, sample_n=300)
        fig = px.scatter(
            pva, x="actual", y="predicted",
            opacity=0.6, color_discrete_sequence=["#6366f1"],
        )
        # Perfect prediction line
        mn, mx = float(pva.min().min()), float(pva.max().max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines", name="Perfect Prediction",
            line=dict(color="#ef4444", dash="dash", width=2),
        ))
        styled_fig(fig, f"Predicted vs Actual — {target_sel.replace('_',' ').title()}", height=370)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Feature importance
        fi = predictor.feature_importance(target=target_sel)
        fig2 = px.bar(
            fi.head(12).sort_values("importance"),
            x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Purples",
        )
        fig2.update_coloraxes(showscale=False)
        styled_fig(fig2, "Feature Importance (Random Forest)", height=370)
        st.plotly_chart(fig2, use_container_width=True)

    # Model comparison table
    st.markdown(section_header("Model Performance Summary"), unsafe_allow_html=True)
    rows = []
    for t in ["engagement_rate", "likes", "comments"]:
        for m in ["ridge", "rf"]:
            mm = predictor.metrics[t][m]
            rows.append({
                "Target":     t.replace("_"," ").title(),
                "Model":      "Random Forest" if m=="rf" else "Ridge Regression",
                "MAE":        mm["mae"],
                "RMSE":       mm["rmse"],
                "R² Score":   mm["r2"],
            })
    summary_df = pd.DataFrame(rows)
    st.dataframe(
        summary_df.style.background_gradient(subset=["R² Score"],
                                              cmap="RdYlGn"),
        use_container_width=True, hide_index=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ANOMALY RADAR
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown(section_header("Anomaly Detection Radar", "MULTI-SIGNAL"),
                unsafe_allow_html=True)

    summary = detector.get_anomaly_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("Total Anomalies",
                                str(summary["total_anomalies"]), icon="⚠️"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Viral Spikes",
                                str(summary["spikes"]), icon="🚀"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Crash Events",
                                str(summary["crashes"]), icon="📉"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("Top Spike Platform",
                                summary["top_spike_platform"], icon="🌐"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Full timeline with anomalies highlighted
    ts_full = engagement_over_time(df, freq="D")
    anomaly_posts = df[df["anomaly_flag"]].copy()
    anomaly_daily = (
        anomaly_posts.set_index("timestamp")
        .resample("D")["total_engagement"].sum()
        .reset_index()
        .rename(columns={"timestamp": "date"})
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_full["date"], y=ts_full["engagement"],
        name="Daily Engagement",
        fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
        line=dict(color="#6366f1", width=1.5),
    ))
    # Anomaly markers
    spikes = df[(df["anomaly_flag"]) & (df["anomaly_type"]=="spike 🚀")]
    crashes= df[(df["anomaly_flag"]) & (df["anomaly_type"]=="crash 📉")]

    if not spikes.empty:
        spike_daily = spikes.set_index("timestamp").resample("D")["total_engagement"].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=spike_daily["timestamp"], y=spike_daily["total_engagement"],
            mode="markers", name="Viral Spike",
            marker=dict(color="#10b981", size=10, symbol="triangle-up"),
        ))
    if not crashes.empty:
        crash_daily = crashes.set_index("timestamp").resample("D")["total_engagement"].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=crash_daily["timestamp"], y=crash_daily["total_engagement"],
            mode="markers", name="Crash Event",
            marker=dict(color="#ef4444", size=10, symbol="triangle-down"),
        ))

    styled_fig(fig, "Engagement Timeline with Anomalies Highlighted", height=360)
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly table
    st.markdown(section_header("Flagged Posts"), unsafe_allow_html=True)
    flagged_df = df[df["anomaly_flag"]][[
        "post_id", "timestamp", "platform", "content_type",
        "total_engagement", "engagement_rate", "z_score",
        "anomaly_type", "anomaly_severity", "signal_count"
    ]].sort_values("z_score", key=abs, ascending=False).head(30)

    # Colour-code anomaly type
    def colour_anomaly(val):
        if "spike" in str(val):
            return "color: #10b981; font-weight:600"
        elif "crash" in str(val):
            return "color: #ef4444; font-weight:600"
        return ""

    st.dataframe(
        flagged_df.style.applymap(colour_anomaly, subset=["anomaly_type"])
                        .background_gradient(subset=["z_score"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True
    )

    # Z-score distribution
    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = px.histogram(
            df, x="z_score", nbins=60,
            color_discrete_sequence=["#6366f1"],
        )
        fig2.add_vline(x=3, line_dash="dash", line_color="#ef4444",
                       annotation_text="+3σ threshold")
        fig2.add_vline(x=-3, line_dash="dash", line_color="#ef4444",
                       annotation_text="-3σ threshold")
        styled_fig(fig2, "Z-Score Distribution of Engagement", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        sev_counts = df[df["anomaly_flag"]]["anomaly_severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        fig3 = px.pie(sev_counts, values="count", names="severity",
                      color_discrete_sequence=["#f59e0b","#ef4444","#8b5cf6","#10b981"],
                      hole=0.45)
        styled_fig(fig3, "Anomaly Severity Breakdown", height=300)
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — AI INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown(section_header("AI-Generated Insights", f"{len(generate_all_insights(df))} INSIGHTS"),
                unsafe_allow_html=True)

    all_insights = generate_all_insights(df)

    # Category filter
    categories_available = sorted(set(ins["category"] for ins in all_insights))
    sel_cat = st.multiselect(
        "Filter by category:", categories_available,
        default=categories_available
    )

    filtered_insights = [i for i in all_insights if i["category"] in sel_cat]

    if not filtered_insights:
        st.warning("No insights for selected categories.")
    else:
        col1, col2 = st.columns(2)
        for idx, ins in enumerate(filtered_insights):
            with (col1 if idx % 2 == 0 else col2):
                st.markdown(insight_card(ins), unsafe_allow_html=True)

    # Quick Summary Stats
    st.markdown(section_header("Follower Tier Analysis", "NANO → MEGA"),
                unsafe_allow_html=True)
    tier_df = follower_tier_analysis(df)
    fig = px.bar(
        tier_df, x="tier", y="avg_eng_rate",
        color="tier", color_discrete_sequence=COLOR_PALETTE,
        text="avg_eng_rate",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(showlegend=False)
    styled_fig(fig, "Avg Engagement Rate by Follower Tier", height=300)
    st.plotly_chart(fig, use_container_width=True)


# ─── DOWNLOAD SECTION ────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(section_header("📥 Export Data"), unsafe_allow_html=True)

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    csv_buf = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Dataset (CSV)",
        data=csv_buf,
        file_name="social_media_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_d2:
    trend_df = detect_trending_posts(df, top_n=20)
    if not trend_df.empty:
        t_csv = trend_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "🔥 Download Trending Posts (CSV)",
            data=t_csv,
            file_name="trending_posts.csv",
            mime="text/csv",
            use_container_width=True,
        )

with col_d3:
    insights_df = pd.DataFrame(generate_all_insights(df))
    ins_csv     = insights_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💡 Download Insights Report (CSV)",
        data=ins_csv,
        file_name="insights_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("<br><br>", unsafe_allow_html=True)
