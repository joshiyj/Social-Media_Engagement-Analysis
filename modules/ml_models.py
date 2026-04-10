"""
modules/ml_models.py
Machine Learning module.

Models:
  1. EngagementPredictor  — Linear Regression + Random Forest to predict
                            likes / comments / engagement_rate
  2. AnomalyDetector      — IQR + Z-score + Isolation Forest flagging

Feature engineering, training, evaluation, and SHAP-style feature importance
are all included.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from sklearn.linear_model  import LinearRegression, Ridge
from sklearn.ensemble      import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics       import (mean_absolute_error, mean_squared_error,
                                   r2_score)
import warnings
warnings.filterwarnings("ignore")


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

CATEGORICAL_COLS = ["platform", "content_type", "category", "weekday"]
NUMERIC_FEATURES = ["hour", "followers", "weekday_num"]
TARGET_OPTIONS   = ["likes", "comments", "engagement_rate"]


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Build a feature matrix from the raw dataframe.
    Returns (feature_df, encoders_dict).
    """
    df = df.copy()

    # ── Encode categoricals ───────────────────────────────────────────────
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # ── Time features ──────────────────────────────────────────────────────
    df["is_weekend"]  = (df["weekday_num"] >= 5).astype(int)
    df["is_primetime"]= df["hour"].isin([18, 19, 20, 21]).astype(int)
    df["log_followers"]= np.log1p(df["followers"])

    # ── Hashtag count ──────────────────────────────────────────────────────
    df["hashtag_count"] = df["hashtags"].str.count(r"\|") + 1

    feature_cols = (
        [f"{c}_enc" for c in CATEGORICAL_COLS]
        + NUMERIC_FEATURES
        + ["is_weekend", "is_primetime", "log_followers", "hashtag_count"]
    )

    return df[feature_cols + TARGET_OPTIONS + ["post_id"]], encoders


# ─── ENGAGEMENT PREDICTOR ─────────────────────────────────────────────────────

class EngagementPredictor:
    """
    Trains both LinearRegression (Ridge) and RandomForest for each target.
    Provides predict(), evaluate(), and feature_importance().
    """

    def __init__(self):
        self.models:   Dict = {}
        self.scalers:  Dict = {}
        self.metrics:  Dict = {}
        self.feature_names: list = []
        self._trained = False

    def fit(self, feature_df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train models for all targets.
        Returns metrics dict.
        """
        feature_cols = [c for c in feature_df.columns
                        if c not in TARGET_OPTIONS + ["post_id"]]
        self.feature_names = feature_cols

        X = feature_df[feature_cols].values

        for target in TARGET_OPTIONS:
            y = feature_df[target].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Scale features for Linear model
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)
            self.scalers[target] = scaler

            # ── Ridge Regression ──────────────────────────────────────────
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_s, y_train)
            y_pred_ridge = ridge.predict(X_test_s)

            # ── Random Forest ─────────────────────────────────────────────
            rf = RandomForestRegressor(
                n_estimators=100, max_depth=8,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)

            self.models[target] = {"ridge": ridge, "rf": rf}

            # ── Metrics ───────────────────────────────────────────────────
            self.metrics[target] = {
                "ridge": {
                    "mae":  round(mean_absolute_error(y_test, y_pred_ridge), 3),
                    "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred_ridge)), 3),
                    "r2":   round(r2_score(y_test, y_pred_ridge), 4),
                },
                "rf": {
                    "mae":  round(mean_absolute_error(y_test, y_pred_rf), 3),
                    "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred_rf)), 3),
                    "r2":   round(r2_score(y_test, y_pred_rf), 4),
                },
                "y_test":       y_test,
                "y_pred_ridge": y_pred_ridge,
                "y_pred_rf":    y_pred_rf,
                "X_test":       X_test,
            }

        self._trained = True
        return self.metrics

    def predict(self, X: np.ndarray, target: str,
                model_type: str = "rf") -> np.ndarray:
        """Predict for new samples."""
        if model_type == "ridge":
            X_s = self.scalers[target].transform(X)
            return self.models[target]["ridge"].predict(X_s)
        return self.models[target]["rf"].predict(X)

    def feature_importance(self, target: str = "engagement_rate") -> pd.DataFrame:
        """Return feature importances from the Random Forest model."""
        rf = self.models[target]["rf"]
        return pd.DataFrame({
            "feature":    self.feature_names,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def predicted_vs_actual(self, target: str,
                            model_type: str = "rf",
                            sample_n: int = 200) -> pd.DataFrame:
        """Return a sampled predicted-vs-actual dataframe for plotting."""
        m      = self.metrics[target]
        y_test = m["y_test"]
        y_pred = m[f"y_pred_{model_type}"]

        idx = np.random.choice(len(y_test), size=min(sample_n, len(y_test)),
                               replace=False)
        return pd.DataFrame({
            "actual":    y_test[idx],
            "predicted": y_pred[idx],
        })


# ─── ANOMALY DETECTOR ─────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Multi-signal anomaly detection:
      1. IQR-based  (statistical outliers in total_engagement)
      2. Z-score    (>3σ from mean)
      3. Isolation Forest (ML-based unsupervised)

    A row is flagged if ≥2 signals agree.
    """

    def __init__(self, contamination: float = 0.03):
        self.contamination = contamination
        self.iso_forest    = IsolationForest(
            contamination=contamination, random_state=42
        )
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds columns: anomaly_iqr, anomaly_zscore, anomaly_iso, anomaly_flag,
        anomaly_type, anomaly_severity.
        """
        df = df.copy()
        eng = df["total_engagement"]

        # ── IQR ───────────────────────────────────────────────────────────
        q1, q3   = eng.quantile(0.25), eng.quantile(0.75)
        iqr      = q3 - q1
        lo, hi   = q1 - 2.0 * iqr, q3 + 2.0 * iqr
        df["anomaly_iqr"] = ~eng.between(lo, hi)

        # ── Z-score ───────────────────────────────────────────────────────
        z = (eng - eng.mean()) / eng.std()
        df["z_score"]       = z.round(3)
        df["anomaly_zscore"]= z.abs() > 3.0

        # ── Isolation Forest ──────────────────────────────────────────────
        feat_cols = ["likes", "comments", "shares", "reach", "total_engagement"]
        X_iso     = df[feat_cols].fillna(0).values
        df["anomaly_iso"] = self.iso_forest.fit_predict(X_iso) == -1
        self._fitted = True

        # ── Consensus flag ────────────────────────────────────────────────
        df["signal_count"] = (
            df["anomaly_iqr"].astype(int)
            + df["anomaly_zscore"].astype(int)
            + df["anomaly_iso"].astype(int)
        )
        df["anomaly_flag"] = df["signal_count"] >= 2

        # ── Type & severity ───────────────────────────────────────────────
        median = eng.median()
        df.loc[df["anomaly_flag"] & (eng > median), "anomaly_type"] = "spike 🚀"
        df.loc[df["anomaly_flag"] & (eng < median), "anomaly_type"] = "crash 📉"
        df["anomaly_type"] = df["anomaly_type"].fillna("normal")

        df["anomaly_severity"] = pd.cut(
            df["z_score"].abs(),
            bins=[0, 1.5, 3.0, 6.0, np.inf],
            labels=["Low", "Medium", "High", "Extreme"]
        )

        return df

    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict:
        """Return counts and key stats about detected anomalies."""
        flagged = df[df["anomaly_flag"]]
        return {
            "total_anomalies": int(flagged["anomaly_flag"].sum()),
            "spikes":  int((flagged["anomaly_type"] == "spike 🚀").sum()),
            "crashes": int((flagged["anomaly_type"] == "crash 📉").sum()),
            "top_spike_platform": (
                flagged[flagged["anomaly_type"] == "spike 🚀"]["platform"].mode()[0]
                if not flagged.empty else "N/A"
            ),
        }
