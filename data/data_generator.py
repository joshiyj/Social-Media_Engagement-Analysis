"""
data_generator.py
Generates a realistic synthetic social media dataset with:
- Temporal patterns (weekday/weekend, hour-of-day effects)
- Platform-specific engagement distributions
- Content type biases
- Hashtag metadata
- Trending spikes and anomalies injected deliberately
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json

# ─── Seed for reproducibility ────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─── Constants ───────────────────────────────────────────────────────────────
PLATFORMS     = ["Instagram", "Twitter", "LinkedIn", "YouTube", "TikTok"]
CONTENT_TYPES = ["Image", "Video", "Text", "Reel/Short", "Carousel"]
CATEGORIES    = ["Tech", "Lifestyle", "Education", "Entertainment", "Finance",
                 "Health", "Travel", "Food", "Sports", "Fashion"]

HASHTAG_POOL = {
    "Tech":          ["#AI", "#MachineLearning", "#Python", "#DataScience", "#CloudComputing", "#DevOps", "#OpenSource"],
    "Lifestyle":     ["#DailyLife", "#Motivation", "#SelfCare", "#Mindfulness", "#Goals"],
    "Education":     ["#Learning", "#StudyTips", "#OnlineCourse", "#Skills", "#Growth"],
    "Entertainment": ["#Trending", "#Viral", "#MustWatch", "#Entertainment", "#Fun"],
    "Finance":       ["#Investing", "#StockMarket", "#Crypto", "#FinancialFreedom", "#Wealth"],
    "Health":        ["#Fitness", "#Wellness", "#Nutrition", "#MentalHealth", "#Yoga"],
    "Travel":        ["#Travel", "#Wanderlust", "#Explore", "#Adventure", "#Backpacking"],
    "Food":          ["#Foodie", "#Recipe", "#Cooking", "#Delicious", "#Vegan"],
    "Sports":        ["#Cricket", "#Football", "#NBA", "#Athletics", "#FitLife"],
    "Fashion":       ["#OOTD", "#Style", "#Fashion", "#Trendy", "#StreetStyle"],
}

# Platform-specific base engagement multipliers
PLATFORM_MULTIPLIERS = {
    "Instagram": {"likes": 1.4, "comments": 0.9, "shares": 0.7, "reach": 1.2},
    "Twitter":   {"likes": 0.8, "comments": 1.2, "shares": 1.5, "reach": 1.0},
    "LinkedIn":  {"likes": 0.9, "comments": 1.3, "shares": 1.1, "reach": 0.8},
    "YouTube":   {"likes": 1.1, "comments": 1.4, "shares": 0.6, "reach": 1.5},
    "TikTok":    {"likes": 1.8, "comments": 1.0, "shares": 1.3, "reach": 1.6},
}

# Content type engagement boost
CONTENT_MULTIPLIERS = {
    "Image":      {"likes": 1.0, "comments": 0.8, "shares": 0.7},
    "Video":      {"likes": 1.3, "comments": 1.2, "shares": 1.5},
    "Text":       {"likes": 0.6, "comments": 1.4, "shares": 0.8},
    "Reel/Short": {"likes": 1.7, "comments": 1.1, "shares": 1.6},
    "Carousel":   {"likes": 1.2, "comments": 1.0, "shares": 1.1},
}

# Best posting hours (engagement multiplier peaks at these hours)
BEST_HOURS = [8, 9, 12, 13, 18, 19, 20, 21]


def _hour_multiplier(hour: int) -> float:
    """Returns a time-of-day engagement multiplier (peaks at BEST_HOURS)."""
    if hour in BEST_HOURS:
        return np.random.uniform(1.3, 1.8)
    elif hour in [6, 7, 10, 11, 14, 15, 22]:
        return np.random.uniform(0.9, 1.2)
    elif hour in [0, 1, 2, 3, 4, 5]:
        return np.random.uniform(0.3, 0.6)   # dead hours
    else:
        return np.random.uniform(0.7, 1.0)


def _weekday_multiplier(weekday: int) -> float:
    """Weekend posts get a small boost on most platforms."""
    if weekday in [5, 6]:   # Saturday, Sunday
        return np.random.uniform(1.1, 1.4)
    elif weekday in [1, 3]:  # Tuesday, Thursday — mid-week peaks
        return np.random.uniform(1.0, 1.2)
    else:
        return np.random.uniform(0.8, 1.1)


def generate_dataset(n_posts: int = 2000,
                     days: int = 180,
                     inject_anomalies: bool = True) -> pd.DataFrame:
    """
    Generate a complete synthetic dataset.

    Parameters
    ----------
    n_posts       : number of posts to generate
    days          : historical window (from today backwards)
    inject_anomalies : inject deliberate viral spikes and crash events
    """

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=days)

    records = []

    for i in range(n_posts):
        # ── Timestamp ──────────────────────────────────────────────────────
        ts = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        hour    = ts.hour
        weekday = ts.weekday()

        # ── Metadata ───────────────────────────────────────────────────────
        platform     = random.choice(PLATFORMS)
        content_type = random.choice(CONTENT_TYPES)
        category     = random.choice(CATEGORIES)
        hashtags     = random.sample(HASHTAG_POOL[category], k=random.randint(1, 4))

        # ── Followers (account size proxy) ─────────────────────────────────
        followers = int(np.random.lognormal(mean=9.5, sigma=1.2))   # 1k – 500k range

        # ── Base engagement ────────────────────────────────────────────────
        p_mult = PLATFORM_MULTIPLIERS[platform]
        c_mult = CONTENT_MULTIPLIERS[content_type]
        t_mult = _hour_multiplier(hour)
        w_mult = _weekday_multiplier(weekday)

        base = followers * 0.03   # baseline 3% engagement

        likes    = max(0, int(base * p_mult["likes"]    * c_mult["likes"]    * t_mult * w_mult * np.random.lognormal(0, 0.4)))
        comments = max(0, int(base * p_mult["comments"] * c_mult["comments"] * t_mult * w_mult * np.random.lognormal(0, 0.5)))
        shares   = max(0, int(base * p_mult["shares"]   * c_mult["shares"]   * t_mult * w_mult * np.random.lognormal(0, 0.6)))
        reach    = max(1, int(followers * p_mult["reach"] * t_mult * np.random.lognormal(0, 0.3)))
        saves    = max(0, int(likes * np.random.uniform(0.05, 0.25)))
        clicks   = max(0, int(reach  * np.random.uniform(0.02, 0.12)))

        total_eng = likes + comments + shares + saves
        eng_rate  = round((total_eng / reach) * 100, 4) if reach > 0 else 0

        records.append({
            "post_id":        f"POST_{i:05d}",
            "timestamp":      ts,
            "date":           ts.date(),
            "hour":           hour,
            "weekday":        ts.strftime("%A"),
            "weekday_num":    weekday,
            "platform":       platform,
            "content_type":   content_type,
            "category":       category,
            "hashtags":       "|".join(hashtags),
            "followers":      followers,
            "likes":          likes,
            "comments":       comments,
            "shares":         shares,
            "saves":          saves,
            "clicks":         clicks,
            "reach":          reach,
            "total_engagement": total_eng,
            "engagement_rate": eng_rate,
        })

    df = pd.DataFrame(records)

    # ── Inject deliberate anomalies ─────────────────────────────────────────
    if inject_anomalies:
        df = _inject_anomalies(df)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly inject viral spikes (10x normal engagement) and
    crash events (0.1x) into ~2% of rows to simulate real-world anomalies.
    """
    n = len(df)
    spike_idx = np.random.choice(n, size=int(n * 0.01), replace=False)
    crash_idx = np.random.choice(
        np.setdiff1d(np.arange(n), spike_idx),
        size=int(n * 0.005), replace=False
    )

    for idx in spike_idx:
        factor = np.random.uniform(8, 20)
        df.loc[idx, ["likes", "comments", "shares"]] = (
            df.loc[idx, ["likes", "comments", "shares"]] * factor
        ).astype(int)
        df.loc[idx, "is_anomaly"]     = "spike"
        df.loc[idx, "anomaly_factor"] = round(factor, 1)

    for idx in crash_idx:
        factor = np.random.uniform(0.05, 0.15)
        df.loc[idx, ["likes", "comments", "shares"]] = (
            df.loc[idx, ["likes", "comments", "shares"]] * factor
        ).astype(int).clip(lower=0)
        df.loc[idx, "is_anomaly"]     = "crash"
        df.loc[idx, "anomaly_factor"] = round(factor, 1)

    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = "normal"
        df["anomaly_factor"] = 1.0
    else:
        df["is_anomaly"]     = df["is_anomaly"].fillna("normal")
        df["anomaly_factor"] = df["anomaly_factor"].fillna(1.0)

    # Recompute totals after anomaly injection
    df["total_engagement"] = df["likes"] + df["comments"] + df["shares"] + df["saves"]
    df["engagement_rate"]  = (df["total_engagement"] / df["reach"].clip(lower=1) * 100).round(4)

    return df


def load_or_generate(path: str = "data/social_media_data.csv",
                     n_posts: int = 2000) -> pd.DataFrame:
    """Load dataset from CSV if it exists, otherwise generate and save."""
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        print(f"[DataLoader] Loaded {len(df)} rows from {path}")
    except FileNotFoundError:
        print(f"[DataLoader] Generating fresh dataset ({n_posts} posts)…")
        df = generate_dataset(n_posts=n_posts)
        df.to_csv(path, index=False)
        print(f"[DataLoader] Saved to {path}")
    return df


if __name__ == "__main__":
    df = generate_dataset(n_posts=2000)
    df.to_csv("social_media_data.csv", index=False)
    print(df.head())
    print(df.dtypes)
    print(f"\nDataset shape: {df.shape}")
    print(f"Anomalies: {df['is_anomaly'].value_counts().to_dict()}")
