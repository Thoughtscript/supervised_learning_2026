import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common.generate_data import CATEGORICAL, DATA_DIR, FEATURES, NUMERIC, REG_TARGET, TARGET

ROOT = DATA_DIR.parent


def load(name):
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def split(df, target=TARGET, seed=0):
    """Train / validation / test = 60 / 20 / 20.

    Validation is for tuning; test is touched once, at the end. Splitting
    happens BEFORE any fitting (scaler, encoder...) so nothing learned from
    validation/test rows leaks into training.
    """
    X, y = df[FEATURES], df[target]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def preprocessor(scale=False, categorical=CATEGORICAL, numeric=NUMERIC):
    """One-hot encode categoricals. handle_unknown='ignore' turns a category
    never seen in training into an all-zero row instead of crashing."""
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", num, numeric),
    ])


def make_pipeline(model, scale=False):
    # The preprocessor lives INSIDE the pipeline: it is fit on training data
    # only and saved with the model, so serving cannot re-implement it wrongly.
    return Pipeline([("prep", preprocessor(scale)), ("model", model)])


def save(model_dir, pipeline, metrics):
    model_dir = Path(model_dir)
    joblib.dump(pipeline, model_dir / "model.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{model_dir.name}] {metrics}")


def load_model(name):
    return joblib.load(ROOT / "models" / name / "model.joblib")


__all__ = ["load", "split", "make_pipeline", "preprocessor", "save", "load_model",
           "TARGET", "REG_TARGET", "FEATURES", "CATEGORICAL", "NUMERIC", "ROOT"]
