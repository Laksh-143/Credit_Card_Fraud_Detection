import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib,os,logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score,accuracy_score,precision_score
from imblearn.over_sampling import BorderlineSMOTE
from config import Config

logger = logging.getLogger(__name__)

def apply_smote(X_train:pd.DataFrame,y_train:pd.Series):
    
    logger.info(f"Before SMOTE: {int(y_train.sum())} fraud / {len(y_train)} total "
                f"({y_train.mean()*100:.2f}%)")
    
    smote = BorderlineSMOTE(
        sampling_strategy = Config.SMOTE_RATIO,
        random_state      = Config.RANDOM_STATE,
        k_neighbors       = 5,
        kind              = "borderline-1",
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
 
    logger.info(f"After  SMOTE: {int(y_res.sum())} fraud / {len(y_res)} total "
                f"({y_res.mean()*100:.2f}%)")
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)

def train_xgboost(X_train:pd.DataFrame,y_train:pd.Series,X_val:pd.DataFrame,y_val:pd.Series)->xgb.XGBClassifier:
    model = xgb.XGBClassifier(**Config.XGB_PARAMS)
    model.fit(X_train,y_train,eval_set = [(X_val,y_val)],verbose=False)
    val_prob = model.predict_proba(X_val)[:,1]
    auc_pr = average_precision_score(y_val,val_prob)
    logger.info(f"XGBoost trained | best_iter={model.best_iteration} | "
                f"val AUC-PR={auc_pr:.4f}")
    return model

def train_lightgbm(X_train:pd.DataFrame,y_train:pd.Series,X_val:pd.DataFrame,y_val:pd.Series)->lgb.LGBMClassifier:
    callbacks = [lgb.early_stopping(Config.EARLY_STOPPING,verbose = False),
                 lgb.log_evaluation(period=1)]
    model = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
    model.fit(X_train,y_train,eval_set = [(X_val,y_val)],callbacks = callbacks)
    val_prob = model.predict_proba(X_val)[:,1]
    auc_pr = average_precision_score(y_val,val_prob)
    logger.info(f"LightGBM trained | best_iter = {model.best_iteration_}|"
                f"val AUC-PR = { auc_pr:.4f}")
    
    return model

    
def ensemble_predict(xgb_model,lgbm_model,X:pd.DataFrame,w_xgb:float = 0.5,w_lgbm:float = 0.5) -> np.ndarray:
    p_xgb = xgb_model.predict_proba(X)[:,1]
    p_lgbm = lgbm_model.predict_proba(X)[:,1]
        
    return w_xgb*p_xgb + w_lgbm*p_lgbm

def cross_validation(X:pd.DataFrame,y:pd.Series,n_splits:int=5)->dict:
    tss = TimeSeriesSplit(n_splits=n_splits)
    scores = {"xgb":[],"lightgbm":[]}
    logger.info(f"TimeSeriesSplit CV with {n_splits} folds")
    
    for fold,(train_idx,val_idx) in enumerate(tss.split(X),1):
        X_tr,X_v = X.iloc[train_idx],X.iloc[val_idx]
        y_tr,y_v = y.iloc[train_idx],y.iloc[val_idx]
        
        if y_v.sum()==0:
            logger.warning(f"fold{fold}: no fraud in val set - skipping")
            continue
        X_tr_sm, y_tr_sm = apply_smote(X_tr,y_tr)
        xgb_m = train_xgboost(X_tr_sm,y_tr_sm,X_v,y_v)
        lgbm_m = train_lightgbm(X_tr_sm,y_tr_sm,X_v,y_v)
        
        xgb_pr = average_precision_score(y_v,xgb_m.predict_proba(X_v)[:,1])
        lgbm_pr = average_precision_score(y_v,lgbm_m.predict_proba(X_v)[:,1])
        
        scores["xgb"].append(xgb_pr)
        scores["lightgbm"].append(lgbm_pr)
        
        logger.info(f"fold{fold} | XGB AUC-PR={xgb_pr:.4f}|LGBM AUC-PR ={lgbm_pr:.4f}")
        
        results = {"xgb_mean": np.mean(scores["xgb"]),
                   "xgb_std": np.std(scores["xgb"]),
                   "lgbm_mean": np.mean(scores["lightgbm"]),
                   "lgbm_std": np.std(scores["lightgbm"])}
        logger.info(f"CV Results → "
                f"XGB: {results['xgb_mean']:.4f} ± {results['xgb_std']:.4f} | "
                f"LGBM: {results['lgbm_mean']:.4f} ± {results['lgbm_std']:.4f}")
        
        return results
    
def optimize_threshold(y_true:pd.Series,y_prob:np.ndarray)->dict:
    thresholds = np.linspace(0.05,0.95,200)
    best_cost = float("inf")
    best_thr = 0.5
        
    costs = []
    for thr in thresholds:
        y_pred = (y_prob>=thr).astype(int)
        fn = int(((y_true==1) & (y_pred==0)).sum())
        fp = int(((y_true==0) & (y_pred==1)).sum())
        cost = fn*Config.COST_FALSE_NEGATIVE+ fp*Config.COST_FALSE_POSITIVE
        costs.append(cost)
        if cost < best_cost:
            best_cost = cost
            best_thr = thr
    logger.info(f"Optimal threshold: {best_thr:.3f} |" f"Estimated cost at threshold : ${best_cost:,.0f}")
        
    os.makedirs(Config.REPORTS_DIR,exist_ok =True)
    plt.figure(figsize=(8,6))
    plt.plot(thresholds, costs, color="steelblue",linewidth = 2)
    plt.axvline(best_thr, color="red",linestyle="--",label=f"Optimal threshold = {best_thr:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Total Business cost($)")
    plt.title("Business Cost vs. Classification Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.REPORTS_DIR,"threshold_cost_curve.png"),dpi=150)
    plt.close()
        
    return {"threshold": best_thr, "min_cost": best_cost}
    
def compute_shap(model,X:pd.DataFrame,model_name:str = "xgb",max_samples:int = 2000)->None:
        
    logger.info(f"Computing SHAP values for {model_name}")
    sample = X.sample(min(max_samples,len(X)),random_state=42)
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
        
    if isinstance(shap_values,list):
        shap_values = shap_values[1]
    os.makedirs(Config.REPORTS_DIR,exist_ok=True)
    plt.figure(figsize=(10,8))
    shap.summary_plot(shap_values,sample,show=False,max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.REPORTS_DIR,f"shap_{model_name}.png"),dpi=150)
    plt.close()
        
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    top_feats = [(sample.columns[i],mean_abs[i]) for i in top_idx]
    logger.info(f"Top-10 SHAP features({model_name}):")
    for feat, val in top_feats:
        logger.info(f" {feat:<35} {val:.4f}")
            
def save_models(xgb_model,lgbm_model,threshold:float)->None:
    os.makedirs(Config.MODELS_DIR,exist_ok=True)
    joblib.dump(xgb_model,os.path.join(Config.MODELS_DIR,"xgboost.pkl"))
    joblib.dump(lgbm_model,os.path.join(Config.MODELS_DIR,"lightgbm.pkl"))
    joblib.dump({"threshold": threshold},
                os.path.join(Config.MODELS_DIR,"threshold.pkl"))
    logger.info(f"Models saved -> {Config.MODELS_DIR}")
        
def load_models() -> tuple:
    xgb_model = joblib.load(os.path.join(Config.MODELS_DIR, "xgboost.pkl"))
    lgbm_model = joblib.load(os.path.join(Config.MODELS_DIR, "lightgbm.pkl"))
    meta       = joblib.load(os.path.join(Config.MODELS_DIR, "threshold.pkl"))
    return xgb_model, lgbm_model, meta["threshold"]
        