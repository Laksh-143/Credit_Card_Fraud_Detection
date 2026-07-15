import numpy as np
import pandas as pd
import joblib, os, logging
from typing import Optional
from config import Config
from feature_engineering import FraudFeatureEngineer
from train import ensemble_predict, load_models

logger = logging.getLogger(__name__)


class FraudPredictor:
    def __init__(self, fe: FraudFeatureEngineer, xgb_model, lgbm_model, threshold: float):
        self.fe = fe
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.threshold = threshold

    @classmethod
    def load(cls, model_dir: str = None) -> "FraudPredictor":
        model_dir = model_dir or Config.MODELS_DIR
        fe = FraudFeatureEngineer.load(model_dir)
        xgb_model, lgbm_model, threshold = load_models()
        logger.info(f"FraudPredictor loaded from {model_dir} | threshold = {threshold:.3f}")
        return cls(fe, xgb_model, lgbm_model, threshold)

    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob < 0.30:  return "Low"
        if prob < 0.50:  return "Medium"
        if prob < 0.75:  return "High"
        return "Critical"

    def _explain_prediction(self, X):
        """Feature-importance-based explanation (replaces SHAP to avoid GPU crash)."""
        try:
            importances = self.xgb_model.feature_importances_

            reasons = []
            for i, col in enumerate(X.columns):
                if i < len(importances) and importances[i] > 0:
                    scaled_val = float(X.iloc[0, i])
                    score = round(float(importances[i] * scaled_val), 4)
                    reasons.append({"feature": col, "shap_value": score})

            reasons.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            return reasons[:5]
        except Exception:
            return []

    def predict_single(self, transaction: dict) -> dict:
        """Score one transaction with user-provided velocity overrides."""

        # Build DataFrame and convert timestamp
        df = pd.DataFrame([transaction])
        df[Config.TIMESTAMP] = pd.to_datetime(df[Config.TIMESTAMP])

        # Ensure velocity columns exist before transform
        for w in Config.VELOCITY_WINDOWS_H:
            for col in [f"txn_count_{w}h", f"amt_sum_{w}h"]:
                if col not in df.columns:
                    df[col] = 0.0

        # Columns we want to override with user-provided values
        override_cols = (
            [f"txn_count_{w}h" for w in Config.VELOCITY_WINDOWS_H]
            + [f"amt_sum_{w}h" for w in Config.VELOCITY_WINDOWS_H]
            + ["is_new_merchant", "is_new_state"]
        )

        # Save user values BEFORE transform overwrites them
        overrides = {}
        for col in override_cols:
            if col in df.columns:
                overrides[col] = float(df[col].iloc[0])

        # Run feature engineering (only ONCE)
        df_feat = self.fe.transform(df)

        # Ensure exact column match with training
        X = df_feat.reindex(columns=self.fe.feature_cols, fill_value=0)

        # Re-inject user-provided values (properly scaled)
        if overrides and hasattr(self.fe.scaler, "feature_names_in_"):
            feat_names = list(self.fe.scaler.feature_names_in_)
            for col, raw_val in overrides.items():
                if col in X.columns and col in feat_names:
                    idx = feat_names.index(col)
                    scaled = (raw_val - self.fe.scaler.mean_[idx]) / self.fe.scaler.scale_[idx]
                    X.at[X.index[0], col] = scaled

        prob = float(ensemble_predict(self.xgb_model, self.lgbm_model, X)[0])
        is_fraud = prob >= self.threshold
        top_reasons = self._explain_prediction(X)

        return {
            "fraud_probability": round(prob, 4),
            "is_fraud":          bool(is_fraud),
            "risk_level":        self._risk_level(prob),
            "threshold_used":    self.threshold,
            "top_reasons":       top_reasons,
        }

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df_feat = self.fe.transform(df_raw)
        X = df_feat.reindex(columns=self.fe.feature_cols, fill_value=0)
        probs = ensemble_predict(self.xgb_model, self.lgbm_model, X)

        df_raw = df_raw.copy()
        df_raw["fraud_probability"] = probs
        df_raw["predicted_fraud"] = (probs >= self.threshold).astype(int)
        df_raw["risk_level"] = [self._risk_level(p) for p in probs]

        return df_raw


# ── FastAPI (only loaded when running uvicorn, not Streamlit) ──

def create_api():
    """Lazy-create FastAPI app to avoid import overhead in Streamlit."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    api = FastAPI(
        title       = "Fraud Detection API",
        description = "Real-time transaction fraud scoring",
        version     = "1.0.0",
    )

    _predictor: Optional[FraudPredictor] = None

    @api.on_event("startup")
    async def startup_event():
        nonlocal _predictor
        try:
            _predictor = FraudPredictor.load()
            logger.info("Predictor loaded successfully at startup")
        except Exception as e:
            logger.error(f"Failed to load predictor: {e}")

    class TransactionRequest(BaseModel):
        trans_date_trans_time: str   = Field(..., example="2020-06-21 12:14:25")
        cc_num:               str   = Field(..., example="4532756279624064")
        merchant:             str   = Field(..., example="fraud_Rippin, Kub and Mann")
        category:             str   = Field(..., example="shopping_net")
        amt:                  float = Field(..., example=149.62)
        gender:               str   = Field(..., example="F")
        city:                 str   = Field(..., example="Orient")
        state:                str   = Field(..., example="WA")
        city_pop:             int   = Field(..., example=859)
        dob:                  str   = Field(..., example="1987-07-25")
        lat:                  float = Field(..., example=44.9592)
        long:                 float = Field(..., example=-85.8847)
        merch_lat:            float = Field(..., example=44.957272)
        merch_long:           float = Field(..., example=-85.953655)

    class PredictionResponse(BaseModel):
        fraud_probability: float
        is_fraud:          bool
        risk_level:        str
        threshold_used:    float
        top_reasons:       list[dict]

    @api.get("/health")
    def health():
        return {
            "status":       "ok",
            "model_loaded": _predictor is not None,
            "threshold":    _predictor.threshold if _predictor else None,
        }

    @api.post("/predict", response_model=PredictionResponse)
    def predict(request: TransactionRequest):
        if _predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        try:
            result = _predictor.predict_single(request.dict())
            return PredictionResponse(**result)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return api
