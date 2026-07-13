import pandas as pd
import numpy as np
import logging
from config import Config
from datetime import timedelta

logger = logging.getLogger(__name__)

def load_data(path: str = None) -> pd.DataFrame:
    """Load raw CSV """
    path = path or Config.DATA_PATH
    logger.info(f"Loading data from {path} ...")
    df = pd.read_csv(path)
 
    df[Config.TIMESTAMP] = pd.to_datetime(df[Config.TIMESTAMP])
    df = df.sort_values(Config.TIMESTAMP).reset_index(drop=True)
 
    logger.info(f"Loaded {len(df):,} rows | "
                f"Fraud rate: {df[Config.TARGET].mean() * 100:.3f}% | "
                f"Date range: {df[Config.TIMESTAMP].min().date()} -> {df[Config.TIMESTAMP].max().date()}")
    return df
def validate_and_clean(df:pd.dataFrame) -> pd.DataFrame:
    
    size = len(df)
    required_columns = [Config.TIMESTAMP,Config.CARD_NUM,Config.AMOUNT,
                        Config.TARGET,Config.CUST_LAT,Config.CUST_LONG,
                        Config.MERCH_LAT,Config.MERCH_LONG]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
 
    # Drop non-positive amounts
    df = df[df[Config.AMOUNT] > 0].copy()
 
    # Valid coordinate ranges: lat ∈ [-90, 90], lon ∈ [-180, 180]
    valid = (
        df[Config.CUST_LAT].between(-90, 90)   &
        df[Config.CUST_LONG].between(-180, 180) &
        df[Config.MERCH_LAT].between(-90, 90)   &
        df[Config.MERCH_LONG].between(-180, 180)
    )
    df = df[valid].copy()
 
    dropped = size - len(df)
    logger.info(f"Validation: dropped {dropped:,} invalid rows | {len(df):,} remaining")
    return df.reset_index(drop=True)

def temporal_split(df: pd.DataFrame):
    
    max_date   = df[Config.TIMESTAMP].max()
    test_start = max_date - timedelta(days=Config.TEST_DAYS)
    val_start  = test_start - timedelta(days=Config.VAL_DAYS)
 
    train = df[df[Config.TIMESTAMP] <  val_start].copy()
    val   = df[(df[Config.TIMESTAMP] >= val_start) &
               (df[Config.TIMESTAMP] <  test_start)].copy()
    test  = df[df[Config.TIMESTAMP] >= test_start].copy()
 
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        logger.info(f"{name:5s}: {len(split):>8,} rows | "
                    f"{split[Config.TIMESTAMP].min().date()} -> "
                    f"{split[Config.TIMESTAMP].max().date()} | "
                    f"Fraud: {split[Config.TARGET].mean()*100:.2f}%")
 
    return train, val, test
 
 
def load_kaggle_split():
    
    train_df = load_data(Config.DATA_PATH)
    test_df  = load_data(Config.TEST_PATH)
    train_df = validate_and_clean(train_df)
    test_df  = validate_and_clean(test_df)
 
    # Create a small validation set from end of train
    val_start  = train_df[Config.TIMESTAMP].max() - timedelta(days=Config.VAL_DAYS)
    val_df     = train_df[train_df[Config.TIMESTAMP] >= val_start].copy()
    train_df   = train_df[train_df[Config.TIMESTAMP] <  val_start].copy()
 
    logger.info(f"Kaggle split -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df