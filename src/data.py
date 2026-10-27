import numpy as np
import pandas as pd
from src.config import TARGET, CATEGORICAL, NON_FEATURES


def load_splits(path):
    """Returns (train, val, test) DataFrames from the cleaned parquet."""
    df = pd.read_parquet(path)
    return tuple(df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))


def for_trees(train, val, test):
    """For XGBoost: NaNs allowed, categoricals as pandas category."""
    cats = [c for c in CATEGORICAL if c in train.columns]
    feats = [c for c in train.columns if c not in NON_FEATURES]
    out = []
    for d in (train, val, test):
        X = d[feats].copy()
        for c in cats:
            X[c] = pd.Categorical(X[c], categories=sorted(train[c].dropna().unique()))
        out.append((X, d[TARGET].to_numpy()))
    return out, feats


def make_nn_data(train, val, test):
    """For all neural models (Keras, fastai, TabNet, Flash). Everything is fit on train only.
    Returns dict of (X_num, X_cat, y), num_cols, cat_cols, cat_cardinalities."""
    cat_cols = [c for c in CATEGORICAL if c in train.columns]
    num_cols = [c for c in train.columns if c not in cat_cols + NON_FEATURES]
    med = train[num_cols].median()
    filled = train[num_cols].fillna(med)
    mu, sd = filled.mean(), filled.std().replace(0, 1)
    maps = {c: {v: i + 1 for i, v in enumerate(train[c].dropna().unique())} for c in cat_cols}  # 0 = unseen
    out = {}
    for name, d in (("train", train), ("val", val), ("test", test)):
        Xn = ((d[num_cols].fillna(med) - mu) / sd).to_numpy("float32")
        Xc = np.column_stack([d[c].map(maps[c]).fillna(0).astype(int) for c in cat_cols])
        out[name] = (Xn, Xc, d[TARGET].to_numpy("float32"))
    return out, num_cols, cat_cols, [len(maps[c]) + 1 for c in cat_cols]
