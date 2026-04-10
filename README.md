# 📊 SocialIQ — Social Media Engagement Intelligence Dashboard

> **Production-grade analytics dashboard** with ML predictions, anomaly detection,
> trending post detection, and auto-generated insights.
> Built for top-tier placement interviews — demonstrates full-stack data engineering.

---

## 🎯 What This Project Demonstrates (For Interviewers)

| Skill Area | What's Shown |
|---|---|
| **Data Engineering** | Synthetic data generation with realistic statistical distributions |
| **Analytics** | KPI computation, time-series aggregation, platform segmentation |
| **Machine Learning** | Ridge Regression + Random Forest with feature engineering |
| **Anomaly Detection** | IQR + Z-score + Isolation Forest (multi-signal consensus) |
| **NLP-style Insights** | Rule-based auto insight generation from data patterns |
| **Visualization** | Plotly (line, bar, pie, heatmap, radar, scatter, box) |
| **System Design** | Modular architecture, caching, filter pipeline |
| **UI/UX** | Custom dark theme, cards, responsive sidebar |
| **Deployment** | Streamlit Cloud ready with config |

---

## 🚀 Quick Start (5 minutes)

### 1. Clone / Download the project
```bash
git clone <your-repo>  # or unzip the folder
cd social_media_dashboard
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the dashboard
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. Done.

---

## 📁 Project Structure

```
social_media_dashboard/
│
├── app.py                         # 🏠 Main Streamlit app (all tabs, sidebar, layout)
├── requirements.txt               # 📦 Python dependencies
├── README.md                      # 📄 This file
│
├── .streamlit/
│   └── config.toml                # 🎨 Dark theme + server config
│
├── assets/
│   └── style.css                  # 🎨 Custom CSS (cards, badges, fonts)
│
├── data/
│   ├── __init__.py
│   ├── data_generator.py          # 🏭 Synthetic dataset factory
│   └── social_media_data.csv      # 📊 Auto-generated on first run
│
├── modules/
│   ├── __init__.py
│   ├── analytics.py               # 📈 KPIs, time-series, heatmaps, trending
│   ├── ml_models.py               # 🤖 ML predictor + anomaly detector
│   └── insights.py                # 💡 Auto insight generator
│
└── exports/                       # 📥 Download output folder
```

---

## 📊 Dataset Schema

The dataset is auto-generated but mirrors real social media APIs.

| Column | Type | Description |
|---|---|---|
| `post_id` | str | Unique post identifier |
| `timestamp` | datetime | Post publish time |
| `platform` | str | Instagram / Twitter / LinkedIn / YouTube / TikTok |
| `content_type` | str | Image / Video / Text / Reel/Short / Carousel |
| `category` | str | Tech / Finance / Health / Travel etc. |
| `hashtags` | str | Pipe-separated hashtags e.g. `#AI\|#Python` |
| `followers` | int | Account follower count (log-normal distribution) |
| `likes` | int | Post likes |
| `comments` | int | Post comments |
| `shares` | int | Post shares/retweets |
| `saves` | int | Post saves/bookmarks |
| `clicks` | int | Link/profile clicks |
| `reach` | int | Number of accounts reached |
| `total_engagement` | int | likes + comments + shares + saves |
| `engagement_rate` | float | total_engagement / reach × 100 |
| `is_anomaly` | str | `normal` / `spike` / `crash` |
| `anomaly_factor` | float | Injection multiplier for anomalies |

**Real data alternatives:**
- [Kaggle: Social Media Influencers Dataset](https://www.kaggle.com/datasets/ramjasmaurya/top-1000-social-media-channels)
- [Kaggle: Instagram Reach Dataset](https://www.kaggle.com/datasets/vikramchaubey/instagram-reach)
- Twitter API v2 (Free tier: 500k tweets/month)

---

## 🧩 Module Breakdown

### `data/data_generator.py`
- Generates 2,000 posts over 180 days with realistic patterns
- **Temporal effects**: Prime-time hours (7–10 PM) boost, weekend lift
- **Platform multipliers**: TikTok gets higher likes/shares; LinkedIn gets more comments
- **Content type bias**: Reels/Video outperform Text
- **Anomaly injection**: 1% viral spikes (×8–20) + 0.5% crash events

### `modules/analytics.py`
- `compute_kpis()` — 10 headline KPI metrics
- `engagement_over_time()` — Daily/weekly time series with rolling avg
- `best_time_heatmap()` — Hour × Weekday pivot for Plotly heatmap
- `detect_trending_posts()` — IQR-based spike detection with explanations
- `hashtag_impact()` — Per-hashtag engagement rate aggregation
- `follower_tier_analysis()` — Nano/Micro/Mid/Macro/Mega segmentation

### `modules/ml_models.py`
**EngagementPredictor:**
- Features: platform, content_type, category, hour, weekday, followers, hashtag_count, is_weekend, is_primetime
- Models: Ridge Regression (with StandardScaler) + Random Forest (100 trees, max_depth=8)
- Outputs: MAE, RMSE, R², predicted-vs-actual scatter data, feature importances

**AnomalyDetector:**
- Signal 1: IQR (Q1 - 2×IQR to Q3 + 2×IQR boundary)
- Signal 2: Z-score > 3σ
- Signal 3: Isolation Forest (contamination=3%)
- **Consensus rule**: flagged if ≥2 signals agree → reduces false positives
- Severity: Low / Medium / High / Extreme based on |z-score|

### `modules/insights.py`
Auto-generates insights by comparing data slices:
- Content type ratios (Video vs Text multiplier)
- Temporal patterns (weekend lift, prime-time edge)
- Platform rankings (highest ER, most shares)
- Hashtag winners (top tag vs average)
- Follower tier winner (Nano often beats Mega)
- Anomaly summary (% of posts flagged)

---

## 🤖 How to Explain the ML in Interviews

**Q: Why Random Forest over a deep learning model?**
> "The dataset is tabular with ~2,000 rows. Random Forest outperforms neural nets on small tabular data, needs no normalization, handles categorical features naturally after label encoding, and produces interpretable feature importances. For production scale (millions of posts), I'd consider XGBoost or LightGBM."

**Q: Why multi-signal anomaly detection?**
> "Each method has different false-positive profiles. IQR catches distributional outliers but misses patterns. Z-score catches deviation from mean but struggles with heavy-tailed distributions. Isolation Forest is unsupervised and model-based. Requiring consensus from ≥2 signals dramatically reduces false positives while maintaining high recall."

**Q: What are the model limitations?**
> "The synthetic data has injected patterns, so R² will be artificially high for some targets. On real social media data, engagement is heavily influenced by external factors (news cycles, algorithm changes) not in the features — so R² of 0.3–0.5 would be realistic and still useful for relative ranking."

---

## 🚀 Deployment

### Option A: Streamlit Community Cloud (Free)
1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as main → Deploy
4. Share the `*.streamlit.app` URL

### Option B: Render (Free Tier)
1. Push to GitHub
2. New Web Service → Python → Build: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Option C: Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🎨 Dashboard Tabs

| Tab | What It Shows |
|---|---|
| 📈 Overview | KPI cards, engagement timeline, rolling avg, platform comparison over time |
| 🌐 Platform Intel | Bar charts, grouped KPIs, multi-metric radar chart |
| 🎨 Content Engine | Pie chart (content share), bar chart (ER by type), category heatmap, hashtag analysis |
| ⏰ Best Time | Hour×Day heatmap, top posting hours, day-of-week bar chart, insight callout |
| 🔥 Trending Posts | Sensitivity slider, scatter plot, trending post cards with explanations |
| 🤖 ML Predictions | Predicted vs actual scatter, feature importance, model comparison table |
| ⚠️ Anomaly Radar | Timeline with spike/crash markers, flagged posts table, Z-score histogram, severity pie |
| 💡 AI Insights | 12+ auto-generated insight cards filterable by category |

---

## 📦 Tech Stack

```
Python 3.11+
├── streamlit        — Web UI framework
├── pandas           — Data manipulation
├── numpy            — Numerical computing
├── plotly           — Interactive charts (line, bar, pie, heatmap, radar, scatter)
└── scikit-learn     — ML (LinearRegression, RandomForest, IsolationForest, LabelEncoder)
```

---

## 🧪 Resume Bullet Points

Copy these for your resume/portfolio:

- Built a **full-stack social media analytics dashboard** in Python (Streamlit + Plotly) with 8 interactive tabs, custom dark UI/CSS, and real-time filter pipeline across 2,000+ simulated posts
- Implemented a **multi-signal anomaly detection system** combining IQR, Z-score (>3σ), and Isolation Forest with consensus voting, reducing false positives vs single-method approaches
- Trained **Ridge Regression and Random Forest** models to predict engagement rate, likes, and comments; visualized predicted-vs-actual and feature importance for interpretability
- Engineered an **automatic insight generator** producing 12+ natural-language findings by comparing content type, timing, platform, hashtag, and audience-tier data slices

---

*Built with ❤️ for campus placements at top-tier tech companies.*
