"""
main.py
──────────────────────────────────────────────────────────────
End-to-end fraud detection pipeline orchestrator.
 
Run:
    python main.py
 
Steps:
    1. Load & validate data
    2. Feature engineering (fit on train, transform val/test)
    3. TimeSeriesSplit cross-validation
    4. Train final XGBoost + LightGBM on full train set
    5. Business-cost threshold optimisation on val set
    6. Full evaluation on held-out test set
    7. SHAP feature importance
    8. Business impact report
    9. Save models and pipeline artifacts
──────────────────────────────────────────────────────────────
"""
 
import logging
import os
import sys
 
import pandas as pd
 
from config import Config
from data_load import (
    load_data,
    validate_and_clean,
    temporal_split,
    load_kaggle_split,
)
from feature_engineering import FraudFeatureEngineer
from Model_comparison import (
    compare_models,
    print_comparison_table,
    plot_model_comparison,
)
from train import (
    apply_smote,
    train_xgboost,
    train_lightgbm,
    cross_validation,
    ensemble_predict,
    optimize_threshold,
    compute_shap,
    save_models,
)
from Evaluate import (
    compute_all_metrics,
    compute_business_impact,
    print_full_report,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_score_distribution,
)
 
 
def setup_logging() -> None:
    os.makedirs("logs", exist_ok=True)
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                "logs/pipeline.log",
                mode     = "w",
                encoding = "utf-8" 
            ),
        ],
    )
 
 
def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
 
    os.makedirs(Config.MODELS_DIR,  exist_ok=True)
    os.makedirs(Config.REPORTS_DIR, exist_ok=True)
 
    logger.info("=" * 60)
    logger.info("  FRAUD DETECTION PIPELINE — START")
    logger.info("=" * 60)
 
    # ── Step 1: Load and validate data ───────────────────────
    logger.info("Step 1: Loading data ...")
 
    # If Kaggle's two pre-split files exist use them
    # Otherwise fall back to single file + temporal split
    if os.path.exists(Config.DATA_PATH) and os.path.exists(Config.TEST_PATH):
        train_df, val_df, test_df = load_kaggle_split()
    elif os.path.exists(Config.DATA_PATH):
        df = load_data(Config.DATA_PATH)
        df = validate_and_clean(df)
        train_df, val_df, test_df = temporal_split(df)
    else:
        logger.error(f"Data not found at: {Config.DATA_PATH}")
        logger.error("Download from: https://www.kaggle.com/datasets/kartik2112/fraud-detection")
        sys.exit(1)
 
    # ── Step 2: Feature engineering ──────────────────────────
    logger.info("Step 2: Feature engineering ...")
 
    fe = FraudFeatureEngineer()
 
    # fit_transform on train — LEARNS encoders and statistics
    train_feat = fe.fit_transform(train_df)
 
    # transform on val/test — APPLIES already-learned transforms
    val_feat   = fe.transform(val_df)
    test_feat  = fe.transform(test_df)
 
    X_train, y_train = fe.get_X_y(train_feat)
    X_val,   y_val   = fe.get_X_y(val_feat)
    X_test,  y_test  = fe.get_X_y(test_feat)
 
    logger.info(
        f"Features: {X_train.shape[1]} | "
        f"Train={len(X_train):,} Val={len(X_val):,} Test={len(X_test):,}"
    )
 
    # Apply SMOTE on training data only
    # Val and test are NEVER resampled — they reflect real distribution
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
 
    # ── Step 3: Algorithm comparison ─────────────────────────
    # Benchmark 5 algorithms on the same data before committing
    # to XGBoost + LightGBM for the full production pipeline
    logger.info("Step 3: Comparing algorithms ...")
 
    results_df, _ = compare_models(
        X_train_sm, y_train_sm,
        X_val,      y_val
    )
 
    print_comparison_table(results_df)
    plot_model_comparison(results_df)
 
    # Save comparison table for README / report
    results_df.to_csv(
        os.path.join(Config.REPORTS_DIR, "model_comparison.csv"),
        index=False
    )
    logger.info("Algorithm comparison saved -> reports/model_comparison.csv")
 
    # ── Step 4: Cross-validation ──────────────────────────────
    # TimeSeriesSplit — each fold trains on past, validates on future
    # Gives mean ± std of AUC-PR across 5 folds (more credible than one number)
    logger.info("Step 4: TimeSeriesSplit cross-validation ...")
 
    X_cv = pd.concat([X_train_sm,
                      pd.DataFrame(X_val.values,  columns=X_val.columns)],
                      ignore_index=True)
    y_cv = pd.concat([y_train_sm,
                      pd.Series(y_val.values)],
                      ignore_index=True)
 
    cv_results = cross_validation(X_cv, y_cv, n_splits=5)
 
    # ── Step 5: Final model training ──────────────────────────
    logger.info("Step 5: Training final XGBoost and LightGBM ...")
 
    xgb_model  = train_xgboost(X_train_sm,  y_train_sm, X_val, y_val)
    lgbm_model = train_lightgbm(X_train_sm, y_train_sm, X_val, y_val)
 
    # ── Step 6: Threshold optimisation on val set ─────────────
    # Standard ML optimises F1 — we optimise for business cost:
    # total_cost = missed_fraud × $250 + false_alarms × $15
    logger.info("Step 6: Optimising classification threshold ...")
 
    val_prob   = ensemble_predict(xgb_model, lgbm_model, X_val)
    thr_result = optimize_threshold(y_val, val_prob)
    threshold  = thr_result["threshold"]
 
    logger.info(f"Optimal threshold: {threshold:.3f} | "
                f"Estimated val cost: ${thr_result['min_cost']:,.0f}")
 
    # ── Step 7: Final evaluation on test set ──────────────────
    logger.info("Step 7: Evaluating on held-out test set ...")
 
    test_prob = ensemble_predict(xgb_model, lgbm_model, X_test)
    test_pred = (test_prob >= threshold).astype(int)
 
    metrics  = compute_all_metrics(y_test, test_prob, threshold)
    business = compute_business_impact(
        y_test, test_pred, test_df[Config.AMOUNT]
    )
 
    print_full_report(metrics, business, cv_results)
 
    # ── Step 8: Visualisations ────────────────────────────────
    logger.info("Step 8: Generating visualisations ...")
 
    plot_confusion_matrix(y_test, test_pred)
    plot_precision_recall_curve(y_test, test_prob, threshold)
    plot_score_distribution(y_test, test_prob)
 
    # ── Step 9: SHAP feature importance ───────────────────────
    logger.info("Step 9: Computing SHAP values ...")
 
    compute_shap(xgb_model,  X_test, model_name="xgboost")
    compute_shap(lgbm_model, X_test, model_name="lightgbm")
 
    # ── Step 10: Save all artifacts ───────────────────────────
    logger.info("Step 10: Saving models and feature pipeline ...")
 
    save_models(xgb_model, lgbm_model, threshold)
    fe.save(Config.MODELS_DIR)
 
    # ── Summary ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  AUC-PR      : {metrics['auc_pr']:.4f}   ← primary metric")
    logger.info(f"  AUC-ROC     : {metrics['auc_roc']:.4f}")
    logger.info(f"  Recall      : {metrics['recall']:.4f}")
    logger.info(f"  Precision   : {metrics['precision']:.4f}")
    logger.info(f"  F1          : {metrics['f1']:.4f}")
    logger.info(f"  Net savings : ${business['net_savings']:,.2f}")
    logger.info(f"  Threshold   : {threshold:.3f}")
    logger.info(f"  Models      → {Config.MODELS_DIR}")
    logger.info(f"  Reports     → {Config.REPORTS_DIR}")
    logger.info("=" * 60)
 
 
if __name__ == "__main__":
    main()