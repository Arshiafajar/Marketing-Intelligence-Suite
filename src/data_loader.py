"""
data_loader.py
Shared data loading and cleaning utilities for the Marketing Intelligence Suite.
All modules import from here to guarantee a single, consistent cleaning pipeline.

Dataset: Online Retail II UCI
Kaggle:  https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
File:    data/raw/online_retail_II.xlsx  (Year 2010-2011 sheet used by default)
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH  = Path(__file__).resolve().parents[1] / "data" / "raw"  / "online_retail_II.xlsx"
PROC_PATH = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_raw(sheet_name: str = "Year 2010-2011") -> pd.DataFrame:
    """
    Load and minimally clean the Online Retail II xlsx file.
    Returns a DataFrame with standardised column names and dtypes.
    """
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"\n\n  Dataset not found at:\n  {RAW_PATH}\n\n"
            "  Download steps:\n"
            "  1. Go to https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci\n"
            "  2. Click 'Download' (free Kaggle account required)\n"
            "  3. Unzip and place 'online_retail_II.xlsx' inside data/raw/\n"
        )

    df = pd.read_excel(RAW_PATH, sheet_name=sheet_name, engine="openpyxl")

    # ── Standardise column names ──────────────────────────────────────────────
    df.columns = df.columns.str.strip()
    rename_map = {
        "Invoice":     "InvoiceNo",
        "StockCode":   "StockCode",
        "Description": "Description",
        "Quantity":    "Quantity",
        "InvoiceDate": "InvoiceDate",
        "Price":       "UnitPrice",
        "Customer ID": "CustomerID",
        "Country":     "Country",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # ── Type coercion ─────────────────────────────────────────────────────────
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"]  = pd.to_numeric(df["CustomerID"], errors="coerce")

    # ── Core cleaning ─────────────────────────────────────────────────────────
    df = df.dropna(subset=["CustomerID"])                                # drop unidentified
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]            # remove cancellations
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]               # positive only
    df["Description"] = df["Description"].fillna("").str.strip().str.title()
    df["CustomerID"]  = df["CustomerID"].astype(int)
    df["TotalPrice"]  = df["Quantity"] * df["UnitPrice"]

    return df.reset_index(drop=True)


def build_rfm(df: pd.DataFrame, holdout_days: int = 90) -> pd.DataFrame:
    """
    Build RFM features using a temporal split to prevent data leakage.

    The dataset is cut at (max_date - holdout_days):
      - Features   -> computed on transactions BEFORE the cutoff
      - WillReturn -> 1 if the customer bought AFTER the cutoff

    Recency in the features therefore describes past behaviour;
    WillReturn describes genuinely future behaviour. No leakage.
    """
    max_date = df["InvoiceDate"].max()
    cutoff   = max_date - pd.Timedelta(days=holdout_days)

    train_df   = df[df["InvoiceDate"] <= cutoff]
    holdout_df = df[df["InvoiceDate"] >  cutoff]

    snapshot = cutoff + pd.Timedelta(days=1)

    rfm = train_df.groupby("CustomerID").agg(
        Recency        = ("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency      = ("InvoiceNo",   "nunique"),
        Monetary       = ("TotalPrice",  "sum"),
        AvgOrderValue  = ("TotalPrice",  "mean"),
        UniqueProducts = ("StockCode",   "nunique"),
        FirstPurchase  = ("InvoiceDate", "min"),
        LastPurchase   = ("InvoiceDate", "max"),
        Country        = ("Country",     lambda x: x.mode()[0]),
    ).reset_index()

    rfm["Tenure"]           = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days + 1
    rfm["FrequencyRecency"] = rfm["Frequency"] / (rfm["Recency"] + 1)
    rfm["MonetaryPerDay"]   = rfm["Monetary"]  / (rfm["Tenure"] + 1)
    rfm["RecencyBucket"]    = pd.cut(
        rfm["Recency"], bins=[0, 30, 90, 180, 365, 9999],
        labels=[5, 4, 3, 2, 1]
    ).astype(int)

    # WillReturn = did this customer buy anything in the holdout window?
    returned        = set(holdout_df["CustomerID"].unique())
    rfm["WillReturn"] = rfm["CustomerID"].isin(returned).astype(int)

    print(f"Temporal split — training up to {cutoff.date()}, holdout = last {holdout_days} days")
    print(f"Customers: {len(rfm):,}  |  WillReturn=1: {rfm['WillReturn'].mean()*100:.1f}%")

    return rfm


def get_descriptions(df: pd.DataFrame, min_chars: int = 10) -> pd.DataFrame:
    """
    Return a deduplicated DataFrame of product descriptions suitable for NLP.
    Each row is one unique (StockCode, Description) pair with purchase count.
    """
    desc = (
        df[df["Description"].str.len() >= min_chars]
        .groupby(["StockCode", "Description"])
        .agg(PurchaseCount=("InvoiceNo", "nunique"))
        .reset_index()
        .sort_values("PurchaseCount", ascending=False)
    )
    return desc
