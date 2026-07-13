
class Config:
 
    # ── Dataset paths ────────────────────────────────────────
    # Kaggle: "Credit Card Transactions Fraud Detection Dataset"
    # https://www.kaggle.com/datasets/kartik2112/fraud-detection
    DATA_PATH   = "data/fraudTrain.csv"
    TEST_PATH   = "data/fraudTest.csv"
    MODELS_DIR  = "saved_models/"
    REPORTS_DIR = "reports/"
 
    # ── Column names ─────────────────────────────────────────
    TIMESTAMP  = "trans_date_trans_time"
    CARD_NUM       = "cc_num"
    MERCHANT   = "merchant"
    CATEGORY   = "category"
    AMOUNT     = "amt"
    TARGET     = "is_fraud"
    CUST_LAT   = "lat"
    CUST_LONG  = "long"
    MERCH_LAT  = "merch_lat"
    MERCH_LONG = "merch_long"
    DOB        = "dob"
    STATE      = "state"
    CITY       = "city"
    CITY_POP   = "city_pop"
    GENDER     = "gender"
    JOB        = "job"
    ZIP        = "zip"
 
    # ── Columns to DROP before training ──────────────────────
    # Reason for each drop is noted
    DROP_COLS = [
        "Unnamed: 0",            # CSV artifact
        "trans_num",             # transaction ID — identifier only
        "unix_time",             # redundant with trans_date_trans_time
        "first",                 # PII — no predictive signal
        "last",                  # PII — no predictive signal
        "street",                # PII — no predictive signal
        "dob",                   # replaced by engineered 'age' feature
        "lat",                   # replaced by engineered 'distance_km' feature
        "long",                  # replaced by engineered 'distance_km' feature
        "merch_lat",             # replaced by engineered 'distance_km' feature
        "merch_long",            # replaced by engineered 'distance_km' feature
        "trans_date_trans_time", # replaced by time features (hour, day, etc.)
        "merchant",              # ~800k unique values — too high cardinality
        "zip",                   # redundant with state + city features
        "job",                   # replaced by engineered 'job_sector' feature
        "city_pop",              # replaced by engineered 'city_pop_bin' feature
        "city",                  # replaced by engineered 'city_freq' feature
    ]
 
    # ── Categorical columns to label encode ──────────────────
    # These are the string columns remaining AFTER feature engineering
    # Note: city_pop_bin and city_freq are NOT here
    #   city_pop_bin → already integer [0,1,2,3,4] from pd.cut
    #   city_freq    → already float (frequency count)
    # Both get standard scaled directly, no label encoding needed
    CAT_COLS = [
        "category",       # 14 unique merchant categories
        "gender",         # M / F
        "state",          # 50 US states
        "job_sector",     # grouped from raw job titles (~10 sectors)
        "category_risk",  # high / medium / low (derived from category)
    ]
 
    # ── Category risk mapping ─────────────────────────────────
    # Online/card-not-present categories = higher fraud risk
    # In-person categories = lower fraud risk
    # Build this from your actual data:
    #   df.groupby("category")["is_fraud"].mean().sort_values(ascending=False)
    CATEGORY_RISK = {
        "shopping_net":   "high",
        "misc_net":       "high",
        "grocery_pos":    "medium",
        "shopping_pos":   "medium",
        "food_dining":    "medium",
        "travel":         "medium",
        "misc_pos":       "medium",
        "health_fitness": "low",
        "entertainment":  "low",
        "gas_transport":  "low",
        "home":           "low",
        "kids_pets":      "low",
        "personal_care":  "low",
        "grocery_net":    "high",
    }
 
    # ── Feature engineering settings ─────────────────────────
    VELOCITY_WINDOWS_H = [1, 6, 24]  # rolling windows in hours
    MIN_TXN_ZSCORE     = 5           # minimum card history before z-score
 
    # ── Temporal split settings ───────────────────────────────
    VAL_DAYS  = 30   # last 30 days of train → validation
    TEST_DAYS = 60   # last 60 days → test set
 
    # ── SMOTE settings ────────────────────────────────────────
    SMOTE_RATIO  = 0.3   # minority:majority ratio after resampling
    RANDOM_STATE = 42
 
    # ── XGBoost hyperparameters ───────────────────────────────
    XGB_PARAMS = dict(
        n_estimators          = 500,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        scale_pos_weight      = 10,      # upweights fraud class
        eval_metric           = "aucpr", # AUC-PR not AUC-ROC
        early_stopping_rounds = 30,
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 0,
        device                = "cuda",
    )
 
    # ── LightGBM hyperparameters ──────────────────────────────
    LGBM_PARAMS = dict(
        n_estimators     = 500,
        max_depth        = 7,
        learning_rate    = 0.05,
        num_leaves       = 63,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        is_unbalance     = True,
        random_state     = 42,
        n_jobs           = -1,
        verbose          = -1,
        device           = "gpu",
    )
 
    EARLY_STOPPING = 30
 
    # ── Business cost threshold optimisation ──────────────────
    # A missed fraud (FN) costs the average fraud transaction amount
    # A false alarm (FP) costs customer friction + support call
    # Tune these numbers to match your actual business context
    COST_FALSE_NEGATIVE = 250   # $ cost of missing a fraud transaction
    COST_FALSE_POSITIVE = 15    # $ cost of blocking a legit transaction