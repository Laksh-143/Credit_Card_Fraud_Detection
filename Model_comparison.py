import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score,
)
import xgboost as xgb
import lightgbm as lgb
 
from config import Config
 
logger = logging.getLogger(__name__)
 
 
# ─────────────────────────────────────────────────────────────
# 1. Define the model zoo
# ─────────────────────────────────────────────────────────────
 
def get_model_zoo() -> dict:
    """
    Returns a dictionary of un-trained model instances.
    Each one will be trained and evaluated identically.
    """
    models = {
 
        "Logistic Regression": LogisticRegression(
            max_iter      = 1000,
            class_weight  = "balanced",   # handles imbalance without SMOTE needed
            random_state  = Config.RANDOM_STATE,
            n_jobs        = -1,
        ),
 
        "Decision Tree": DecisionTreeClassifier(
            max_depth     = 8,
            class_weight  = "balanced",
            random_state  = Config.RANDOM_STATE,
        ),
 
        "Random Forest": RandomForestClassifier(
            n_estimators  = 300,
            max_depth     = 10,
            class_weight  = "balanced",
            random_state  = Config.RANDOM_STATE,
            n_jobs        = -1,
        ),
 
        "XGBoost": xgb.XGBClassifier(
            n_estimators      = Config.XGB_PARAMS["n_estimators"],
            max_depth         = Config.XGB_PARAMS["max_depth"],
            learning_rate     = Config.XGB_PARAMS["learning_rate"],
            subsample         = Config.XGB_PARAMS["subsample"],
            colsample_bytree  = Config.XGB_PARAMS["colsample_bytree"],
            scale_pos_weight  = Config.XGB_PARAMS["scale_pos_weight"],
            eval_metric       = "aucpr",
            random_state      = Config.RANDOM_STATE,
            n_jobs            = -1,
            verbosity         = 0,
            device            = "cuda",
        ),
 
        "LightGBM": lgb.LGBMClassifier(
            n_estimators      = Config.LGBM_PARAMS["n_estimators"],
            max_depth         = Config.LGBM_PARAMS["max_depth"],
            learning_rate     = Config.LGBM_PARAMS["learning_rate"],
            num_leaves        = Config.LGBM_PARAMS["num_leaves"],
            subsample         = Config.LGBM_PARAMS["subsample"],
            colsample_bytree  = Config.LGBM_PARAMS["colsample_bytree"],
            is_unbalance      = True,
            random_state      = Config.RANDOM_STATE,
            n_jobs            = -1,
            verbose           = -1,
            device           = "gpu",
        ),
    }
    return models
 
 
# ─────────────────────────────────────────────────────────────
# 2. Train + evaluate every model, collect results
# ─────────────────────────────────────────────────────────────
 
def compare_models(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame,   y_val: pd.Series,
                   threshold: float = 0.5) -> pd.DataFrame:
    """
    Train every model in the zoo on (X_train, y_train),
    evaluate on (X_val, y_val), return a comparison dataframe.
 
    Note: X_train/y_train should already be SMOTE-resampled
    before calling this function (see train.py: apply_smote).
    """
    models  = get_model_zoo()
    results = []
    fitted_models = {}
 
    logger.info(f"Comparing {len(models)} algorithms ...")
    logger.info(f"Train size: {len(X_train):,} | Val size: {len(X_val):,}")
 
    for name, model in models.items():
        logger.info(f"Training {name} ...")
        start = time.time()
 
        model.fit(X_train, y_train)
        train_time = time.time() - start
 
        # Predict probabilities
        val_prob = model.predict_proba(X_val)[:, 1]
        val_pred = (val_prob >= threshold).astype(int)
 
        # Metrics
        auc_pr  = average_precision_score(y_val, val_prob)
        auc_roc = roc_auc_score(y_val, val_prob)
        f1      = f1_score(y_val, val_pred, zero_division=0)
        prec    = precision_score(y_val, val_pred, zero_division=0)
        rec     = recall_score(y_val, val_pred, zero_division=0)
 
        results.append({
            "Model":          name,
            "AUC-PR":         round(auc_pr, 4),
            "AUC-ROC":        round(auc_roc, 4),
            "F1":             round(f1, 4),
            "Precision":      round(prec, 4),
            "Recall":         round(rec, 4),
            "Train Time (s)": round(train_time, 2),
        })
 
        fitted_models[name] = model
 
        logger.info(f"  {name:<22} AUC-PR={auc_pr:.4f} | AUC-ROC={auc_roc:.4f} | "
                    f"F1={f1:.4f} | Time={train_time:.1f}s")
 
    results_df = (
        pd.DataFrame(results)
        .sort_values("AUC-PR", ascending=False)
        .reset_index(drop=True)
    )
 
    return results_df, fitted_models
 
 
# ─────────────────────────────────────────────────────────────
# 3. Print formatted comparison table
# ─────────────────────────────────────────────────────────────
 
def print_comparison_table(results_df: pd.DataFrame) -> None:
    print("\n" + "=" * 75)
    print("  MODEL COMPARISON — ALGORITHM BENCHMARK")
    print("=" * 75)
    print(results_df.to_string(index=False))
    print("=" * 75)
 
    best = results_df.iloc[0]
    print(f"\n  Best model by AUC-PR: {best['Model']} ({best['AUC-PR']:.4f})")
    print(f"  This is the primary metric for imbalanced fraud detection.\n")
 
 
# ─────────────────────────────────────────────────────────────
# 4. Visualisation
# ─────────────────────────────────────────────────────────────
 
def plot_model_comparison(results_df: pd.DataFrame,
                          save_path: str = "reports/model_comparison.png") -> None:
    """
    Two-panel chart:
      Left  : AUC-PR per model (the metric that matters)
      Right : AUC-PR vs training time (efficiency trade-off)
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(results_df)))
 
    # Left: bar chart of AUC-PR
    sorted_df = results_df.sort_values("AUC-PR", ascending=True)
    axes[0].barh(sorted_df["Model"], sorted_df["AUC-PR"], color=colors)
    axes[0].set_xlabel("AUC-PR (higher is better)")
    axes[0].set_title("Model Performance — AUC-PR Comparison")
    for i, v in enumerate(sorted_df["AUC-PR"]):
        axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
 
    # Right: efficiency scatter (AUC-PR vs training time)
    axes[1].scatter(results_df["Train Time (s)"], results_df["AUC-PR"],
                    s=150, c=colors, edgecolors="black", linewidth=1)
    for _, row in results_df.iterrows():
        axes[1].annotate(row["Model"],
                         (row["Train Time (s)"], row["AUC-PR"]),
                         textcoords="offset points", xytext=(8, 4), fontsize=9)
    axes[1].set_xlabel("Training Time (seconds)")
    axes[1].set_ylabel("AUC-PR")
    axes[1].set_title("Performance vs. Training Cost")
    axes[1].set_xscale("log")
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Comparison chart saved → {save_path}")
 
 
# ─────────────────────────────────────────────────────────────
# 5. Standalone test
# ─────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
 
    from data_load import load_data, validate_and_clean, temporal_split
    from feature_engineering import FraudFeatureEngineer
    from train import apply_smote
 
    df = load_data()
    df = validate_and_clean(df)
    train_df, val_df, test_df = temporal_split(df)
 
    fe = FraudFeatureEngineer()
    train_feat = fe.fit_transform(train_df)
    val_feat   = fe.transform(val_df)
 
    X_train, y_train = fe.get_X_y(train_feat)
    X_val,   y_val   = fe.get_X_y(val_feat)
 
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
 
    results_df, fitted_models = compare_models(X_train_sm, y_train_sm, X_val, y_val)
    print_comparison_table(results_df)
    plot_model_comparison(results_df)
 
    results_df.to_csv("reports/model_comparison.csv", index=False)
    print("Results saved → reports/model_comparison.csv")
