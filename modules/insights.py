"""
modules/insights.py
Automatic Insight Generator.

Compares slices of the dataset and produces natural-language insight strings
that are displayed in the dashboard's "AI Insight Panel".

Every function returns a list of dicts: [{icon, headline, detail, category}]
"""

import pandas as pd
import numpy as np
from typing import List, Dict


def _pct(a: float, b: float) -> str:
    """Format percentage change from b to a."""
    if b == 0:
        return "∞%"
    chg = (a - b) / b * 100
    sign = "+" if chg >= 0 else ""
    return f"{sign}{chg:.0f}%"


# ─── 1. CONTENT TYPE INSIGHTS ────────────────────────────────────────────────

def content_type_insights(df: pd.DataFrame) -> List[Dict]:
    insights = []
    ct = df.groupby("content_type")["engagement_rate"].mean()

    best  = ct.idxmax()
    worst = ct.idxmin()
    ratio = ct[best] / max(ct[worst], 0.001)

    insights.append({
        "icon":     "🎬",
        "headline": f"{best} content drives {ratio:.1f}× more engagement than {worst}",
        "detail":   (f"Average engagement rate: {best} = {ct[best]:.2f}%, "
                     f"{worst} = {ct[worst]:.2f}%."),
        "category": "Content Type",
    })

    # Reels/Short video check
    if "Reel/Short" in ct.index and "Text" in ct.index:
        mult = ct["Reel/Short"] / max(ct["Text"], 0.001)
        insights.append({
            "icon":     "📱",
            "headline": f"Reels/Shorts outperform plain text by {mult:.1f}×",
            "detail":   "Short-form video captures algorithm boosts and higher completion rates.",
            "category": "Content Type",
        })

    return insights


# ─── 2. TEMPORAL INSIGHTS ────────────────────────────────────────────────────

def temporal_insights(df: pd.DataFrame) -> List[Dict]:
    insights = []

    # Weekend vs weekday
    wknd = df[df["weekday_num"] >= 5]["engagement_rate"].mean()
    wkdy = df[df["weekday_num"] <  5]["engagement_rate"].mean()
    if wknd > wkdy:
        insights.append({
            "icon":     "📅",
            "headline": f"Weekend posts outperform weekday posts ({_pct(wknd, wkdy)})",
            "detail":   f"Weekend avg ER: {wknd:.2f}% vs Weekday: {wkdy:.2f}%.",
            "category": "Timing",
        })
    else:
        insights.append({
            "icon":     "📅",
            "headline": f"Weekday posts drive more engagement ({_pct(wkdy, wknd)})",
            "detail":   f"Weekday avg ER: {wkdy:.2f}% vs Weekend: {wknd:.2f}%.",
            "category": "Timing",
        })

    # Prime time check (7 PM – 10 PM)
    prime = df[df["hour"].isin([19, 20, 21, 22])]["engagement_rate"].mean()
    off   = df[df["hour"].isin([0, 1, 2, 3, 4, 5])]["engagement_rate"].mean()
    insights.append({
        "icon":     "🌙",
        "headline": f"7 PM–10 PM posts get {_pct(prime, off)} more engagement than late-night posts",
        "detail":   f"Prime-time ER: {prime:.2f}% vs Late-night: {off:.2f}%.",
        "category": "Timing",
    })

    # Best single hour
    best_hour = df.groupby("hour")["engagement_rate"].mean().idxmax()
    insights.append({
        "icon":     "⏰",
        "headline": f"Hour {best_hour}:00 is the single best time to post",
        "detail":   (f"Posts published at {best_hour}:00 average "
                     f"{df[df['hour']==best_hour]['engagement_rate'].mean():.2f}% ER."),
        "category": "Timing",
    })

    return insights


# ─── 3. PLATFORM INSIGHTS ────────────────────────────────────────────────────

def platform_insights(df: pd.DataFrame) -> List[Dict]:
    insights = []
    pt = df.groupby("platform")["engagement_rate"].mean().sort_values(ascending=False)

    top_p  = pt.index[0]
    low_p  = pt.index[-1]

    insights.append({
        "icon":     "🏆",
        "headline": f"{top_p} delivers the highest average engagement rate",
        "detail":   (f"{top_p}: {pt[top_p]:.2f}% | {low_p}: {pt[low_p]:.2f}%. "
                     f"Consider focusing more posts on {top_p}."),
        "category": "Platform",
    })

    # Shares leader
    share_leader = df.groupby("platform")["shares"].mean().idxmax()
    insights.append({
        "icon":     "🔁",
        "headline": f"{share_leader} drives the most shares (virality engine)",
        "detail":   (f"Avg shares on {share_leader}: "
                     f"{df[df['platform']==share_leader]['shares'].mean():.0f} per post."),
        "category": "Platform",
    })

    return insights


# ─── 4. HASHTAG INSIGHTS ─────────────────────────────────────────────────────

def hashtag_insights(df: pd.DataFrame) -> List[Dict]:
    insights = []

    exploded = df.assign(
        hashtag=df["hashtags"].str.split("|")
    ).explode("hashtag")

    top_tag = exploded.groupby("hashtag")["engagement_rate"].mean().idxmax()
    top_er  = exploded.groupby("hashtag")["engagement_rate"].mean().max()

    base_er = df["engagement_rate"].mean()
    lift    = top_er / max(base_er, 0.001)

    insights.append({
        "icon":     "#️⃣",
        "headline": f"'{top_tag}' is the highest-performing hashtag ({lift:.1f}× avg ER)",
        "detail":   (f"Posts tagged {top_tag} average {top_er:.2f}% ER vs "
                     f"dataset average of {base_er:.2f}%."),
        "category": "Hashtags",
    })

    # Hashtag count analysis
    df2 = df.copy()
    df2["htag_count"] = df2["hashtags"].str.count(r"\|") + 1
    optimal = df2.groupby("htag_count")["engagement_rate"].mean().idxmax()
    insights.append({
        "icon":     "🏷️",
        "headline": f"Posts with {optimal} hashtag(s) earn peak engagement",
        "detail":   "Beyond the optimal count, hashtag stuffing may reduce reach.",
        "category": "Hashtags",
    })

    return insights


# ─── 5. FOLLOWER TIER INSIGHTS ────────────────────────────────────────────────

def follower_tier_insights(df: pd.DataFrame) -> List[Dict]:
    insights = []
    bins   = [0, 10_000, 50_000, 200_000, 1_000_000, np.inf]
    labels = ["Nano", "Micro", "Mid", "Macro", "Mega"]
    df2    = df.copy()
    df2["tier"] = pd.cut(df2["followers"], bins=bins, labels=labels)

    tier_er = df2.groupby("tier", observed=False)["engagement_rate"].mean()
    top_tier = tier_er.idxmax()

    insights.append({
        "icon":     "👥",
        "headline": f"{top_tier}-influencer accounts have the highest engagement rate",
        "detail":   (f"ER: {tier_er[top_tier]:.2f}%. "
                     "Micro-influencers often outperform mega accounts due to niche audiences."),
        "category": "Audience",
    })

    return insights


# ─── 6. ANOMALY INSIGHTS ─────────────────────────────────────────────────────

def anomaly_insights(df: pd.DataFrame) -> List[Dict]:
    insights = []

    if "anomaly_flag" not in df.columns:
        return insights

    flagged = df[df["anomaly_flag"]]
    pct_anomaly = len(flagged) / len(df) * 100

    insights.append({
        "icon":     "⚠️",
        "headline": f"{pct_anomaly:.1f}% of posts show anomalous engagement patterns",
        "detail":   (f"{(flagged['anomaly_type']=='spike 🚀').sum()} spikes and "
                     f"{(flagged['anomaly_type']=='crash 📉').sum()} crashes detected."),
        "category": "Anomalies",
    })

    if not flagged.empty:
        spike_platform = (
            flagged[flagged["anomaly_type"] == "spike 🚀"]["platform"].mode()
        )
        if not spike_platform.empty:
            insights.append({
                "icon":     "🚀",
                "headline": f"Most viral spikes occur on {spike_platform.iloc[0]}",
                "detail":   "Viral events concentrate on platforms with higher algorithmic amplification.",
                "category": "Anomalies",
            })

    return insights


# ─── MASTER INSIGHT GENERATOR ────────────────────────────────────────────────

def generate_all_insights(df: pd.DataFrame) -> List[Dict]:
    """Run all insight generators and return combined list."""
    all_insights = []
    generators = [
        content_type_insights,
        temporal_insights,
        platform_insights,
        hashtag_insights,
        follower_tier_insights,
        anomaly_insights,
    ]
    for gen in generators:
        try:
            all_insights.extend(gen(df))
        except Exception as e:
            pass  # Gracefully skip failing generators

    return all_insights
