"""
modules/analytics.py
Core analytics engine.

Provides:
  - KPI summaries
  - Platform-level breakdowns
  - Content type performance
  - Best-time-to-post heatmap data
  - Trending post detection (IQR-based spike detection)
  - Hashtag impact analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


# ─── 1. KPI SUMMARY ──────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> Dict:
    """Return top-level KPI numbers for the metric cards."""
    total_posts      = len(df)
    total_likes      = int(df["likes"].sum())
    total_comments   = int(df["comments"].sum())
    total_shares     = int(df["shares"].sum())
    total_reach      = int(df["reach"].sum())
    avg_eng_rate     = round(float(df["engagement_rate"].mean()), 2)
    total_saves      = int(df["saves"].sum())
    total_clicks     = int(df["clicks"].sum())
    best_platform    = df.groupby("platform")["engagement_rate"].mean().idxmax()
    best_content     = df.groupby("content_type")["engagement_rate"].mean().idxmax()

    return {
        "total_posts":    total_posts,
        "total_likes":    total_likes,
        "total_comments": total_comments,
        "total_shares":   total_shares,
        "total_reach":    total_reach,
        "avg_eng_rate":   avg_eng_rate,
        "total_saves":    total_saves,
        "total_clicks":   total_clicks,
        "best_platform":  best_platform,
        "best_content":   best_content,
    }


# ─── 2. TIME-SERIES AGGREGATION ──────────────────────────────────────────────

def engagement_over_time(df: pd.DataFrame,
                         freq: str = "D") -> pd.DataFrame:
    """
    Aggregate total_engagement by date.
    freq: 'D' = daily, 'W' = weekly
    """
    ts = (
        df.set_index("timestamp")
          .resample(freq)["total_engagement"]
          .sum()
          .reset_index()
          .rename(columns={"timestamp": "date", "total_engagement": "engagement"})
    )
    # 7-day rolling average
    ts["rolling_avg"] = ts["engagement"].rolling(7, min_periods=1).mean().round(0)
    return ts


def engagement_by_platform_over_time(df: pd.DataFrame,
                                     freq: str = "W") -> pd.DataFrame:
    """Weekly engagement broken down by platform (for multi-line chart)."""
    ts = (
        df.set_index("timestamp")
          .groupby("platform")
          .resample(freq)["total_engagement"]
          .sum()
          .reset_index()
          .rename(columns={"timestamp": "date"})
    )
    return ts


# ─── 3. PLATFORM BREAKDOWN ───────────────────────────────────────────────────

def platform_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-platform aggregated KPIs."""
    grp = df.groupby("platform").agg(
        posts          = ("post_id",          "count"),
        avg_likes      = ("likes",            "mean"),
        avg_comments   = ("comments",         "mean"),
        avg_shares     = ("shares",           "mean"),
        avg_eng_rate   = ("engagement_rate",  "mean"),
        total_reach    = ("reach",            "sum"),
    ).round(2).reset_index()
    return grp


# ─── 4. CONTENT TYPE PERFORMANCE ─────────────────────────────────────────────

def content_type_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregated engagement metrics by content type."""
    grp = df.groupby("content_type").agg(
        count          = ("post_id",         "count"),
        avg_likes      = ("likes",           "mean"),
        avg_comments   = ("comments",        "mean"),
        avg_shares     = ("shares",          "mean"),
        avg_eng_rate   = ("engagement_rate", "mean"),
        total_eng      = ("total_engagement","sum"),
    ).round(2).reset_index()
    return grp


# ─── 5. BEST-TIME-TO-POST HEATMAP ────────────────────────────────────────────

def best_time_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pivot table: rows = hour (0-23), cols = weekday (Mon-Sun),
    values = avg engagement rate — ready for a Plotly heatmap.
    """
    DAY_ORDER  = ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"]

    pivot = (
        df.groupby(["hour", "weekday"])["engagement_rate"]
          .mean()
          .unstack("weekday")
          .reindex(columns=DAY_ORDER)
          .fillna(0)
          .round(3)
    )
    return pivot


def best_posting_hours(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return the top N hours sorted by mean engagement rate."""
    hourly = (
        df.groupby("hour")["engagement_rate"]
          .mean()
          .reset_index()
          .sort_values("engagement_rate", ascending=False)
          .head(top_n)
          .round(3)
    )
    return hourly


# ─── 6. TRENDING POST DETECTION ──────────────────────────────────────────────

def detect_trending_posts(df: pd.DataFrame,
                          multiplier: float = 2.5,
                          top_n: int = 10) -> pd.DataFrame:
    """
    A post is 'trending' if its total_engagement exceeds:
        median + multiplier * IQR
    Returns top_n trending posts with explanation strings.
    """
    q1  = df["total_engagement"].quantile(0.25)
    q3  = df["total_engagement"].quantile(0.75)
    iqr = q3 - q1
    threshold = df["total_engagement"].median() + multiplier * iqr

    trending = df[df["total_engagement"] > threshold].copy()

    if trending.empty:
        return pd.DataFrame()

    # Compute how many standard deviations above mean
    mu  = df["total_engagement"].mean()
    std = df["total_engagement"].std()
    trending["z_score"]   = ((trending["total_engagement"] - mu) / std).round(2)
    trending["spike_pct"] = (((trending["total_engagement"] - mu) / mu) * 100).round(1)

    # Human-readable explanation
    def _explain(row):
        parts = []
        if row["content_type"] in ["Video", "Reel/Short"]:
            parts.append("video content drove higher shares")
        if row["hour"] in [19, 20, 21]:
            parts.append("posted during prime-time hours")
        if row["weekday"] in ["Saturday", "Sunday"]:
            parts.append("weekend audience boost")
        if row["is_anomaly"] == "spike":
            parts.append(f"viral event (×{row['anomaly_factor']} normal)")
        if not parts:
            parts.append("organic engagement spike")
        return "; ".join(parts).capitalize() + "."

    trending["explanation"] = trending.apply(_explain, axis=1)
    trending = trending.sort_values("total_engagement", ascending=False).head(top_n)
    return trending.reset_index(drop=True)


# ─── 7. HASHTAG IMPACT ───────────────────────────────────────────────────────

def hashtag_impact(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Explode the pipe-separated hashtag column, then aggregate
    average engagement rate per hashtag.
    """
    exploded = df.assign(
        hashtag=df["hashtags"].str.split("|")
    ).explode("hashtag")

    impact = (
        exploded.groupby("hashtag").agg(
            posts        = ("post_id",          "count"),
            avg_eng_rate = ("engagement_rate",  "mean"),
            avg_likes    = ("likes",            "mean"),
            avg_shares   = ("shares",           "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values("avg_eng_rate", ascending=False)
        .head(top_n)
    )
    return impact


# ─── 8. CATEGORY PERFORMANCE ─────────────────────────────────────────────────

def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Engagement breakdown by content category."""
    return (
        df.groupby("category").agg(
            posts        = ("post_id",          "count"),
            avg_eng_rate = ("engagement_rate",  "mean"),
            avg_likes    = ("likes",            "mean"),
            avg_comments = ("comments",         "mean"),
            avg_shares   = ("shares",           "mean"),
        )
        .round(3)
        .sort_values("avg_eng_rate", ascending=False)
        .reset_index()
    )


# ─── 9. FOLLOWER TIER SEGMENTATION ───────────────────────────────────────────

def follower_tier_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin followers into tiers (Nano / Micro / Mid / Macro / Mega)
    and compare average engagement rates across tiers.
    """
    bins   = [0, 10_000, 50_000, 200_000, 1_000_000, np.inf]
    labels = ["Nano (<10K)", "Micro (10K-50K)",
              "Mid (50K-200K)", "Macro (200K-1M)", "Mega (1M+)"]

    df = df.copy()
    df["tier"] = pd.cut(df["followers"], bins=bins, labels=labels)
    return (
        df.groupby("tier", observed=False).agg(
            accounts     = ("post_id",          "count"),
            avg_eng_rate = ("engagement_rate",  "mean"),
            avg_likes    = ("likes",            "mean"),
        )
        .round(3)
        .reset_index()
    )
