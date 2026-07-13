import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import logging, os
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, confusion_matrix,
    classification_report, f1_score, precision_score, recall_score,
)
from config import Config
 
logger = logging.getLogger(__name__)
 
 
def compute_all_metrics(y_true: pd.Series, y_prob: np.ndarray,
                        threshold: float) -> dict:
    """Compute full suite of evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
 
    auc_pr  = average_precision_score(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)
    ks      = _ks_statistic(y_true, y_prob)
    f1      = f1_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
 
    return {
        "auc_pr":      auc_pr,
        "auc_roc":     auc_roc,
        "ks_statistic":ks,
        "f1":          f1,
        "precision":   prec,
        "recall":      rec,
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "threshold":   threshold,
    }
 
 
def _ks_statistic(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """
    KS (Kolmogorov-Smirnov) statistic — max separation between the
    cumulative fraud score distribution and the cumulative legit
    score distribution. Used in banking risk models.
    Range [0, 1]. Higher = better model separation.
    """
    df = pd.DataFrame({"prob": y_prob, "label": y_true})
    df = df.sort_values("prob")
    df["cum_fraud"] = (df["label"] == 1).cumsum() / max((y_true == 1).sum(), 1)
    df["cum_legit"] = (df["label"] == 0).cumsum() / max((y_true == 0).sum(), 1)
    return float((df["cum_fraud"] - df["cum_legit"]).abs().max())
 
 
def compute_business_impact(y_true: pd.Series, y_pred: np.ndarray,
                             amounts: pd.Series) -> dict:
    """
    Translate model performance into dollar terms.
    This is what management cares about, not F1.
    """
    tp_mask = (y_true == 1) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
 
    fraud_caught_dollars = float(amounts[tp_mask].sum())
    fraud_missed_dollars = float(amounts[fn_mask].sum())
    fp_cost              = int(fp_mask.sum()) * Config.COST_FALSE_POSITIVE
    net_savings          = fraud_caught_dollars - fp_cost
 
    return {
        "fraud_caught_dollars": fraud_caught_dollars,
        "fraud_missed_dollars": fraud_missed_dollars,
        "n_false_positives":    int(fp_mask.sum()),
        "false_positive_cost":  fp_cost,
        "net_savings":          net_savings,
        "fraud_catch_rate_pct": float(tp_mask.sum()) / max((y_true == 1).sum(), 1) * 100,
    }
 
 
def print_full_report(metrics: dict, business: dict, cv_results: dict) -> None:
    """Print formatted evaluation report to stdout and logger."""
    div = "=" * 60
 
    report = f"""
{div}
FRAUD DETECTION PIPELINE — EVALUATION REPORT
{div}
 
── MODEL PERFORMANCE ──────────────────────────────────────
  AUC-PR    (primary): {metrics['auc_pr']:.4f}
  AUC-ROC            : {metrics['auc_roc']:.4f}
  KS Statistic       : {metrics['ks_statistic']:.4f}
  F1 Score           : {metrics['f1']:.4f}
  Precision          : {metrics['precision']:.4f}
  Recall             : {metrics['recall']:.4f}
  Threshold used     : {metrics['threshold']:.3f}
 
── CONFUSION MATRIX ────────────────────────────────────────
  True Positives (fraud caught)   : {metrics['tp']:>8,}
  False Negatives (fraud missed)  : {metrics['fn']:>8,}
  False Positives (legit blocked) : {metrics['fp']:>8,}
  True Negatives (legit passed)   : {metrics['tn']:>8,}
 
── BUSINESS IMPACT ─────────────────────────────────────────
  Fraud caught ($)           : ${business['fraud_caught_dollars']:>12,.2f}
  Fraud missed ($)           : ${business['fraud_missed_dollars']:>12,.2f}
  False positive cost ($)    : ${business['false_positive_cost']:>12,.2f}
  Net savings ($)            : ${business['net_savings']:>12,.2f}
  Fraud catch rate           : {business['fraud_catch_rate_pct']:.1f}%
 
── CROSS-VALIDATION (TimeSeriesSplit) ──────────────────────
  XGBoost  AUC-PR: {cv_results['xgb_mean']:.4f} ± {cv_results['xgb_std']:.4f}
  LightGBM AUC-PR: {cv_results['lgbm_mean']:.4f} ± {cv_results['lgbm_std']:.4f}
 
{div}
"""
    print(report)
    logger.info(report)
 
    # Save report to file
    os.makedirs(Config.REPORTS_DIR, exist_ok=True)
    with open(os.path.join(Config.REPORTS_DIR, "evaluation_report.txt"), "w",encoding = "utf-8") as f:
        f.write(report)
 
 
def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Legit", "Predicted Fraud"],
        yticklabels=["Actual Legit", "Actual Fraud"],
    )
    plt.title("Confusion Matrix — Fraud Detection Ensemble")
    plt.tight_layout()
    os.makedirs(Config.REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.REPORTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    logger.info("Saved confusion_matrix.png")
 
 
def plot_precision_recall_curve(y_true: pd.Series, y_prob: np.ndarray,
                                  threshold: float) -> None:
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
 
    # Find point on curve closest to our chosen threshold
    idx = np.argmin(np.abs(thresholds - threshold))
 
    plt.figure(figsize=(8, 5))
    plt.plot(rec, prec, color="steelblue", linewidth=2,
             label=f"PR Curve (AUC-PR = {auc_pr:.3f})")
    plt.scatter(rec[idx], prec[idx], color="red", s=80, zorder=5,
                label=f"Operating point (thr={threshold:.2f})")
    plt.axhline(y_true.mean(), linestyle="--", color="grey",
                label=f"Baseline (fraud rate = {y_true.mean():.3f})")
    plt.xlabel("Recall (Fraud Catch Rate)")
    plt.ylabel("Precision (Accuracy when flagging)")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.REPORTS_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close()
    logger.info("Saved precision_recall_curve.png")
 
 
def plot_score_distribution(y_true: pd.Series, y_prob: np.ndarray) -> None:
    """Distribution of fraud scores for legit vs. fraud transactions."""
    plt.figure(figsize=(9, 4))
    plt.hist(y_prob[y_true == 0], bins=60, alpha=0.6,
             color="steelblue", label="Legitimate", density=True)
    plt.hist(y_prob[y_true == 1], bins=60, alpha=0.6,
             color="crimson", label="Fraud", density=True)
    plt.xlabel("Predicted Fraud Probability")
    plt.ylabel("Density")
    plt.title("Score Distribution: Fraud vs. Legitimate Transactions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.REPORTS_DIR, "score_distribution.png"), dpi=150)
    plt.close()
    logger.info("Saved score_distribution.png")