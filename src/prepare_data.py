"""Download the sample, clean it, split it, and save data/processed/loans_clean.parquet.

Run from the repo root:  python -m src.prepare_data
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import SEED, CATEGORICAL

URL = "https://raw.githubusercontent.com/tiwvis/MLBD-Projects/main/data_sets/LoansData_sample.csv.gz"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "LoansData_sample.csv.gz")
OUT = os.path.join(ROOT, "data", "processed", "loans_clean.parquet")

DROP = ["out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
        "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
        "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
        "last_fico_range_high", "last_fico_range_low", "debt_settlement_flag", "policy_code",
        "pymnt_plan", "loan_status", "id", "member_id", "url", "desc", "title", "emp_title",
        "zip_code", "funded_amnt", "funded_amnt_inv"]
DROP_PREFIXES = ("hardship_", "settlement_", "sec_app_", "orig_projected", "deferral", "payment_plan")


def main():
    source = RAW if os.path.exists(RAW) else URL
    print("reading:", source)
    df = pd.read_csv(source, compression="gzip", low_memory=False)

    # target: keep loans with a final outcome
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
    df["default"] = df["loan_status"].isin(["Charged Off", "Default"]).astype(int)

    # drop leakage columns, identifiers, free text
    drop = DROP + [c for c in df.columns if c.startswith(DROP_PREFIXES)]
    df = df.drop(columns=[c for c in drop if c in df.columns])

    # dates -> credit history length (issue_dt kept only for reference)
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_hist_years"] = (df["issue_dt"] - earliest).dt.days / 365.25
    df = df.drop(columns=["issue_d", "earliest_cr_line"])

    # term and employment length
    df["term_months"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0])
    el = df["emp_length"].astype(str)
    df["emp_length_yrs"] = np.where(el.str.contains(r"10\+"), 10,
                           np.where(el.str.contains("< 1"), 0,
                                    pd.to_numeric(el.str.extract(r"(\d+)")[0], errors="coerce")))
    df = df.drop(columns=["term", "emp_length"])

    # other text columns become numeric, except the categoricals
    cats = [c for c in CATEGORICAL if c in df.columns]
    for c in df.select_dtypes("object").columns:
        if c not in cats:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace("%", "", regex=False), errors="coerce")

    # stratified random 70/15/15 split
    idx_tr, idx_rest = train_test_split(df.index, test_size=0.30, stratify=df["default"], random_state=SEED)
    idx_va, idx_te = train_test_split(idx_rest, test_size=0.50, stratify=df.loc[idx_rest, "default"], random_state=SEED)
    df["split"] = "train"
    df.loc[idx_va, "split"] = "val"
    df.loc[idx_te, "split"] = "test"

    # drop columns that are >50% missing, judged on the training rows only
    miss = df.loc[df["split"] == "train"].isna().mean()
    df = df.drop(columns=miss[miss > 0.5].index.tolist())

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print("saved:", OUT, df.shape)
    print(df.groupby("split")["default"].agg(["size", "mean"]))


if __name__ == "__main__":
    main()