import numpy as np
import pandas as pd
import joblib,os,logging
from typing import Optional
import shap
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from config import Config
from feature_engineering import FraudFeatureEngineer
from train import ensemble_predict,load_models

logger = logging.getLogger(__name__)


class FraudPredictor:
    def __init__(self,fe: FraudFeatureEngineer,xgb_model,lgbm_model,threshold:float):
        
        self.fe = fe
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.threshold = threshold
        self._explainer = shap.TreeExplainer(xgb_model)
        
    @classmethod
    def load(cls,model_dir:str=None) ->"FraudPredictor":
        model_dir = model_dir or Config.MODELS_DIR
        fe = FraudFeatureEngineer.load(model_dir)
        xgb_model, lgbm_model , threshold = load_models()
        logger.info(f"FraudPredictor loaded from {model_dir} | threshold = {threshold:.3f}")
        return cls(fe,xgb_model,lgbm_model,threshold)
    
    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob < 0.30:  return "Low"
        if prob < 0.50:  return "Medium"
        if prob < 0.75:  return "High"
        return "Critical"

    
    def predict_single(self, transaction:dict) ->dict:
        df = pd.DataFrame([transaction])
        df[Config.TIMESTAMP] = pd.to_datetime(df[Config.TIMESTAMP])
        
        for w in Config.VELOCITY_WINDOWS_H:
            for col in [f"txn_count_{w}h",f"amt_sum_{w}h"]:
                if col not in df.columns:
                    df[col] =0.0
        df_feat = self.fe.transform(df)
        X = df_feat[[c for c in self.fe.feature_cols if c in df_feat.columns]]
        prob = float(ensemble_predict(self.xgb_model,self.lgbm_model,X)[0])
        is_fraud = prob >= self.threshold
        
        shap_vals = self._explainer.shap_values(X)
        if isinstance(shap_vals,list):
            shap_vals = shap_vals[1]
        shap_vals = shap_vals[0]
        
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
        top_reasons = [{"feature":X.columns[i],"shap_value":round(float(shap_vals[i]),4)} for i in top_idx]
        
        return {
            "fraud_probability": round(prob, 4),
            "is_fraud":          bool(is_fraud),
            "risk_level":        self._risk_level(prob),
            "threshold_used":    self.threshold,
            "top_reasons":       top_reasons,
        }
        
    def predict_batch(self, df_raw:pd.DataFrame)->pd.DataFrame:
        df_feat = self.fe.transform(df_raw)
        X = df_feat[[c for c in self.fe.feature_cols if c in df_feat.columns]]
        probs = ensemble_predict(self.xgb_model,self.lgbm_model)
        
        df_raw = df_raw.copy()
        df_raw["fraud_probability"] = probs
        df_raw["predicted_fraud"] = (probs >= self.threshold).astype(int)
        df_raw["risk_level"] = [self._risk_level(p) for p in probs]
        
        return df_raw
    
app = FastAPI(
    title       = "Fraud Detection API",
    description = "Real-time transaction fraud scoring — XGBoost + LightGBM ensemble",
    version     = "1.0.0",
)
 
# Load predictor once at startup
_predictor: Optional[FraudPredictor] = None
 
@app.on_event("startup")
async def startup_event():
    global _predictor
    try:
        _predictor = FraudPredictor.load()
        logger.info("Predictor loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load predictor: {e}")
 
 
# ── Request / Response schemas ───────────────────────────────
 
class TransactionRequest(BaseModel):
    trans_date_trans_time : str   = Field(..., example="2020-06-21 12:14:25")
    cc_num                : str   = Field(..., example="4532756279624064")
    merchant              : str   = Field(..., example="fraud_Rippin, Kub and Mann")
    category              : str   = Field(..., example="shopping_net")
    amt                   : float = Field(..., example=149.62)
    gender                : str   = Field(..., example="F")
    city                  : str   = Field(..., example="Orient")
    state                 : str   = Field(..., example="WA")
    city_pop              : int   = Field(..., example=859)
    dob                   : str   = Field(..., example="1987-07-25")
    lat                   : float = Field(..., example=44.9592)
    long                  : float = Field(..., example=-85.8847)
    merch_lat             : float = Field(..., example=44.957272)
    merch_long            : float = Field(..., example=-85.953655)
 
 
class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]
 
 
class PredictionResponse(BaseModel):
    fraud_probability : float
    is_fraud          : bool
    risk_level        : str
    threshold_used    : float
    top_reasons       : list[dict]
 
 
class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total       : int
    flagged     : int
 
 
# ── Endpoints ────────────────────────────────────────────────
 
@app.get("/health")
def health():
    return {
        "status":    "ok",
        "model_loaded": _predictor is not None,
        "threshold": _predictor.threshold if _predictor else None,
    }
 
 
@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = _predictor.predict_single(request.dict())
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        predictions = [_predictor.predict_single(t.dict()) for t in request.transactions]
        flagged     = sum(1 for p in predictions if p["is_fraud"])
        return BatchResponse(
            predictions=[PredictionResponse(**p) for p in predictions],
            total       = len(predictions),
            flagged     = flagged,
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
