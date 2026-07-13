"""
feature_engineering.py
─────────────────────────────────────────────────────────────
FraudFeatureEngineer — complete feature pipeline in one class.

Features built:
  Base        : haversine distance, age, city_pop_bin, city_freq,
                job_sector, category_risk
  Temporal    : hour, day, month, is_night, is_weekend,
                cyclical encoding, days since last transaction
  Velocity    : per-card rolling count and spend over 1h / 6h / 24h
  Behavioural : amount z-score vs own history, new merchant flag,
                new state flag, amount vs category mean ratio

Two key methods:
  fit_transform(train_df) → learns encoders and stats from training data
  transform(val/test_df)  → applies already-learned transforms only

FIXES from previous project:
  1. LabelEncoder bug — old code used one variable `le` in a loop,
     so only the last column's encoder survived. Fixed by storing
     one encoder per column in a dict: self.label_encoders[col]

  2. Coordinate validation — old mask used OR instead of AND,
     dropping valid rows and keeping invalid ones. Fixed in data_loader.

  3. Random split — replaced with temporal split in data_loader.

NEW vs previous project:
  - city_pop binned into 5 tiers instead of raw number
  - city replaced by frequency encoding (city_freq)
  - job collapsed into ~10 job sectors
  - category_risk tier added
  - velocity features (1h/6h/24h rolling per card)
  - behavioural features (z-score, new merchant, new state)
─────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib, logging, os
from config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Job sector mapping — defined here as a module constant
# Collapses hundreds of job titles into ~10 sectors
# Add more mappings if your dataset has different job titles
# ─────────────────────────────────────────────────────────────
JOB_SECTOR_MAP = {
    # Technology
    "Software Engineer":      "tech",
    "Data Scientist":         "tech",
    "Systems Analyst":        "tech",
    "IT Manager":             "tech",
    "Web Developer":          "tech",
    "Database Administrator": "tech",

    # Healthcare
    "Nurse":                  "healthcare",
    "Doctor":                 "healthcare",
    "Pharmacist":             "healthcare",
    "Physiotherapist":        "healthcare",
    "Surgeon":                "healthcare",
    "Dentist":                "healthcare",

    # Finance
    "Accountant":             "finance",
    "Financial Advisor":      "finance",
    "Banker":                 "finance",
    "Actuary":                "finance",
    "Economist":              "finance",
    "Auditor":                "finance",

    # Education
    "Teacher":                "education",
    "Professor":              "education",
    "Lecturer":               "education",
    "Librarian":              "education",

    # Legal
    "Lawyer":                 "legal",
    "Solicitor":              "legal",
    "Barrister":              "legal",
    "Judge":                  "legal",

    # Engineering
    "Civil Engineer":         "engineering",
    "Mechanical Engineer":    "engineering",
    "Electrical Engineer":    "engineering",
    "Chemical Engineer":      "engineering",

    # Retail / Service
    "Sales Manager":          "retail",
    "Shop Manager":           "retail",
    "Customer Service":       "retail",

    # Creative / Media
    "Journalist":             "media",
    "Designer":               "media",
    "Writer":                 "media",
    "Photographer":           "media",

    # Trades
    "Electrician":            "trades",
    "Plumber":                "trades",
    "Carpenter":              "trades",
    "Mechanic":               "trades",
}


class FraudFeatureEngineer:

    def __init__(self):
        # Stores one LabelEncoder per categorical column
        # KEY FIX: not a single variable that gets overwritten in a loop
        self.label_encoders : dict[str, LabelEncoder] = {}

        self.scaler          = StandardScaler()

        # Stores city → frequency count, learned from training data only
        self.city_freq_map   : dict[str, int] = {}

        self.feature_cols    : list[str] = []
        self._fitted         = False

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run on TRAINING data only.
        Builds all features AND fits encoders/scaler on training stats.
        """
        logger.info("Feature engineering: fit_transform started ...")

        df = self._build_features(df, fit=True)
        df = self._encode_and_scale(df, fit=True)

        # Store feature column names (everything except target and card number)
        self.feature_cols = [
            c for c in df.columns
            if c not in (Config.TARGET, Config.CARD_NUM)
        ]

        self._fitted = True
        logger.info(f"fit_transform complete — {len(self.feature_cols)} features, "
                    f"{len(df):,} rows")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run on VAL or TEST data.
        Applies encoders and scaler that were fitted on training data.
        Never re-learns anything — uses stored mappings only.
        """
        if not self._fitted:
            raise RuntimeError(
                "Pipeline not fitted. Call fit_transform on training data first."
            )
        logger.info("Feature engineering: transform started ...")
        df = self._build_features(df, fit=False)
        df = self._encode_and_scale(df, fit=False)
        return df

    def get_X_y(self, df: pd.DataFrame):
        """
        Separate feature matrix X from target column y.
        Use after fit_transform or transform.
        """
        feature_cols = [c for c in self.feature_cols if c in df.columns]
        X = df[feature_cols]
        y = df[Config.TARGET] if Config.TARGET in df.columns else None
        return X, y

    def save(self, directory: str) -> None:
        """Save all fitted objects so inference can reload them."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.label_encoders, os.path.join(directory, "label_encoders.pkl"))
        joblib.dump(self.scaler,         os.path.join(directory, "scaler.pkl"))
        joblib.dump(self.city_freq_map,  os.path.join(directory, "city_freq_map.pkl"))
        joblib.dump(self.feature_cols,   os.path.join(directory, "feature_cols.pkl"))
        logger.info(f"Feature engineer saved → {directory}")

    @classmethod
    def load(cls, directory: str) -> "FraudFeatureEngineer":
        """Reload a previously saved feature engineer for inference."""
        fe = cls()
        fe.label_encoders = joblib.load(os.path.join(directory, "label_encoders.pkl"))
        fe.scaler         = joblib.load(os.path.join(directory, "scaler.pkl"))
        fe.city_freq_map  = joblib.load(os.path.join(directory, "city_freq_map.pkl"))
        fe.feature_cols   = joblib.load(os.path.join(directory, "feature_cols.pkl"))
        fe._fitted        = True
        return fe

    # ─────────────────────────────────────────────────────────
    # Internal pipeline — step 1: build new feature columns
    # ─────────────────────────────────────────────────────────

    def _build_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()

        # Parse timestamp if it came in as a string (happens during inference)
        df[Config.TIMESTAMP] = pd.to_datetime(df[Config.TIMESTAMP])

        df = df.sort_values(Config.TIMESTAMP).reset_index(drop=True)
        df = self._base_features(df, fit)
        df = self._temporal_features(df)
        df = self._velocity_features(df)
        df = self._behavioural_features(df)
        return df

    # ── Base features ─────────────────────────────────────────

    def _base_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Creates features from static transaction and customer info.
        fit parameter controls city_freq encoding (learn vs apply).
        """

        # ── Haversine distance: home → merchant ───────────────
        # Fraud signal: large distance = card possibly stolen and
        # used far from home
        df["distance_km"] = self._haversine(
            df[Config.CUST_LAT].values,
            df[Config.CUST_LONG].values,
            df[Config.MERCH_LAT].values,
            df[Config.MERCH_LONG].values,
        )

        # ── Age from date of birth ─────────────────────────────
        # Different age groups have different fraud vulnerability
        dob       = pd.to_datetime(df[Config.DOB], errors="coerce")
        trans_day = df[Config.TIMESTAMP].dt.normalize()
        df["age"] = (
            (trans_day - dob).dt.days / 365.25
        ).clip(0, 120).fillna(35)

        # ── city_pop_bin ───────────────────────────────────────
        # Raw population (e.g. 859, 34000, 1200000) is hard for
        # a tree model to split on meaningfully. Binning into 5
        # named tiers makes the signal explicit.
        # Labels are integers 0-4 so NO label encoding needed later
        df["city_pop_bin"] = pd.cut(
            df[Config.CITY_POP],
            bins   = [0, 1_000, 10_000, 100_000, 1_000_000, float("inf")],
            labels = [0, 1, 2, 3, 4],     # tiny, small, medium, large, metro
        ).astype(float)

        # ── city_freq: frequency encoding ─────────────────────
        # Too many unique cities for label encoding.
        # Replace each city with how often it appears in the dataset.
        # High-frequency city = high population = different fraud pattern.
        #
        # fit=True  → learn city→count mapping from training data, store it
        # fit=False → apply stored mapping (never relearn on val/test)
        if fit:
            self.city_freq_map = df[Config.CITY].value_counts().to_dict()

        df["city_freq"] = (
            df[Config.CITY]
            .map(self.city_freq_map)
            .fillna(1)         # cities not seen in training get frequency 1
            .astype(float)
        )

        # ── job_sector: group job titles into sectors ──────────
        # Raw 'job' has hundreds of unique values like "Geophysicist",
        # "Traffic warden" which label encoding treats as arbitrary numbers.
        # Grouping into ~10 meaningful sectors gives the model
        # a learnable signal about occupational fraud patterns.
        df["job_sector"] = (
            df[Config.JOB]
            .map(JOB_SECTOR_MAP)
            .fillna("other")   # any job title not in the map → "other"
        )

        # ── category_risk: risk tier from merchant category ───
        # Online card-not-present categories have higher fraud rates.
        # Maps category string → "high" / "medium" / "low"
        # fillna("medium") handles any categories not in our mapping
        df["category_risk"] = (
            df[Config.CATEGORY]
            .map(Config.CATEGORY_RISK)
            .fillna("medium")
        )

        return df

    # ── Temporal features ─────────────────────────────────────

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from the transaction timestamp.
        Fraud spikes at night, on weekends, and in certain months.
        """
        ts = df[Config.TIMESTAMP]

        df["hour"]        = ts.dt.hour.astype(int)
        df["day_of_week"] = ts.dt.dayofweek.astype(int)   # 0=Mon, 6=Sun
        df["month"]       = ts.dt.month.astype(int)
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

        # 11pm to 6am = high risk window
        df["is_night"]    = (
            (df["hour"] >= 23) | (df["hour"] <= 6)
        ).astype(int)

        # Cyclical encoding — prevents model treating 23h and 0h as far apart
        # Without this: hour 23 and hour 0 differ by 23 numerically
        # With this:    their sin/cos values are very close (1 hour apart on a circle)
        df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Days since this card last transacted
        # Fraud signal: very short gap = card being rapidly drained
        df = df.sort_values([Config.CARD_NUM, Config.TIMESTAMP])
        prev_ts = df.groupby(Config.CARD_NUM)[Config.TIMESTAMP].shift(1)
        df["days_since_last_txn"] = (
            (df[Config.TIMESTAMP] - prev_ts)
            .dt.total_seconds() / 86400
        ).fillna(-1)    # -1 = first transaction for this card (no history)

        return df

    # ── Velocity features ─────────────────────────────────────

    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each transaction, count how many times the same card
        was used in the past 1h, 6h, and 24h.
        Also sums the total spend in those windows.

        Fraud signal: stolen cards get used many times rapidly.
        This is consistently the highest-importance feature group.

        Leakage prevention:
          closed='left' → window is [T-W, T) which EXCLUDES the current
          transaction from its own window. No future data included.
        """
        df = df.copy()

        # Unique row identifier for aligning results back
        df["_row_id"] = np.arange(len(df))

        for window_h in Config.VELOCITY_WINDOWS_H:
            count_col  = f"txn_count_{window_h}h"
            sum_col    = f"amt_sum_{window_h}h"
            window_str = f"{window_h}h"

            count_map: dict[int, float] = {}
            sum_map:   dict[int, float] = {}

            for _, group in df.groupby(Config.CARD_NUM, sort=False):
                grp = group.sort_values(Config.TIMESTAMP)

                # Set timestamp as index for pandas time-based rolling
                g_indexed = grp.set_index(Config.TIMESTAMP)

                rolling = g_indexed[Config.AMOUNT].rolling(
                    window     = window_str,
                    closed     = "left",       # exclude current transaction
                    min_periods= 0,
                )

                counts  = rolling.count().fillna(0).values
                sums    = rolling.sum().fillna(0).values
                row_ids = grp["_row_id"].values

                for i, rid in enumerate(row_ids):
                    count_map[rid] = counts[i]
                    sum_map[rid]   = sums[i]

            df[count_col] = df["_row_id"].map(count_map)
            df[sum_col]   = df["_row_id"].map(sum_map)

            logger.info(f"Velocity {window_h}h | "
                        f"mean count={df[count_col].mean():.2f} | "
                        f"mean spend=${df[sum_col].mean():.2f}")

        df = df.drop(columns=["_row_id"])
        return df

    # ── Behavioural features ──────────────────────────────────

    def _behavioural_features(self, df: pd.DataFrame) -> pd.DataFrame:
   
        df = df.copy()
        df = df.sort_values([Config.CARD_NUM, Config.TIMESTAMP]).reset_index(drop=True)

        # ── Amount z-score vs own card history ──────────────────
        # expanding().mean().shift(1) = mean of all PAST transactions
        # shift(1) moves the window back by 1 so current transaction
        # is never included in its own mean/std calculation
        card_grp = df.groupby(Config.CARD_NUM)[Config.AMOUNT]

        card_past_mean  = card_grp.transform(lambda x: x.expanding().mean().shift(1))
        card_past_std   = card_grp.transform(lambda x: x.expanding().std().shift(1))
        card_past_count = df.groupby(Config.CARD_NUM).cumcount()  # 0 = first transaction

        df["amt_zscore"] = np.where(
            card_past_count >= Config.MIN_TXN_ZSCORE,
            (df[Config.AMOUNT] - card_past_mean) / (card_past_std + 1e-8),
            0.0
        )
        df["amt_zscore"] = df["amt_zscore"].fillna(0.0)

        # ── Is new merchant for this card ───────────────────────
        # cumcount() on (card, merchant) group:
        # 0 = first time this card used this merchant = new merchant
        # Since df is sorted by [CARD, TIMESTAMP], temporal order is respected
        df["is_new_merchant"] = (
            df.groupby([Config.CARD_NUM, Config.MERCHANT]).cumcount() == 0
        ).astype(int)

        # ── Is new state for this card ───────────────────────────
        df["is_new_state"] = (
            df.groupby([Config.CARD_NUM, Config.STATE]).cumcount() == 0
        ).astype(int)

        # ── Amount vs category mean ──────────────────────────────
        cat_mean = df.groupby(Config.CATEGORY)[Config.AMOUNT].transform("mean")
        df["amt_vs_cat_mean"] = (
            df[Config.AMOUNT] / cat_mean.replace(0, 1)
        ).astype(float)

        logger.info(
            f"Behavioural features done | "
            f"new_merchant_rate={df['is_new_merchant'].mean():.3f} | "
            f"new_state_rate={df['is_new_state'].mean():.3f}"
        )

        return df

    # ─────────────────────────────────────────────────────────
    # Internal pipeline — step 2: encode and scale
    # ─────────────────────────────────────────────────────────

    def _encode_and_scale(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        1. Drop raw columns (now replaced by engineered features)
        2. Label encode string categorical columns
        3. Standard scale all numeric features
        """

        # ── Drop raw columns ──────────────────────────────────
        drop = [c for c in Config.DROP_COLS if c in df.columns]
        df   = df.drop(columns=drop)

        # ── Label encode categorical columns ──────────────────
        # KEY FIX: self.label_encoders is a DICT, not a single variable.
        # Old (broken):
        #   for col in cols:
        #       le = LabelEncoder()   ← same variable, overwritten each loop
        #       le.fit(df[col])       ← only last encoder survives
        #
        # Correct:
        #   for col in cols:
        #       le = LabelEncoder()
        #       self.label_encoders[col] = le   ← stored separately per column

        for col in Config.CAT_COLS:
            if col not in df.columns:
                continue

            df[col] = df[col].astype(str).fillna("unknown")

            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le           # stored in dict

            else:
                le    = self.label_encoders[col]        # retrieved from dict
                known = set(le.classes_)

                # Handle categories not seen during training
                # (e.g. a new state in test data that wasn't in train)
                df[col] = df[col].apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])

        # Note: city_pop_bin → already integer 0-4, no encoding needed
        # Note: city_freq    → already float, no encoding needed
        # Both will be picked up by StandardScaler below

        # ── Standard scale all numeric features ───────────────
        # Exclude target and card number (not features)
        exclude   = [Config.TARGET, Config.CARD_NUM]
        num_cols  = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

        # Fill any remaining NaN with column median before scaling
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        if fit:
            df[num_cols] = self.scaler.fit_transform(df[num_cols].astype(float))
        else:
            scale_cols = list(self.scaler.feature_names_in_)
            for c in scale_cols:
                if c not in df.columns:
                    df[c] = 0.0
            df[scale_cols] = df[scale_cols].fillna(0).astype(float)
            df[scale_cols] = self.scaler.transform(df[scale_cols])

        return df

    # ─────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _haversine(lat1: np.ndarray, lon1: np.ndarray,
                   lat2: np.ndarray, lon2: np.ndarray,
                   R: float = 6371.0) -> np.ndarray:
        """
        Vectorised Haversine formula — great-circle distance in km.
        R = 6371.0 km (Earth's mean radius)
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
        return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        