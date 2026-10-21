from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common.utils import DATA_DIR, save

CAT, NUM = ["dow"], ["lag_1", "lag_7", "lag_14", "roll_mean_7"]


def features(df):
    """A linear model (Ridge) extrapolates the trend that tree models cannot.
    Lag features only look BACKWARD (shift) so no future value leaks in.
    Day-of-week is one-hot encoded: Monday=0..Sunday=6 is not an ordered scale."""
    out = pd.DataFrame(index=df.index)
    out["dow"] = df["date"].dt.dayofweek.astype(str)
    out["lag_1"] = df["sales"].shift(1)
    out["lag_7"] = df["sales"].shift(7)
    out["lag_14"] = df["sales"].shift(14)
    out["roll_mean_7"] = df["sales"].shift(1).rolling(7).mean()
    return out


def build():
    prep = ColumnTransformer([("onehot", OneHotEncoder(handle_unknown="ignore"), CAT),
                              ("num", "passthrough", NUM)])
    return Pipeline([("prep", prep), ("model", Ridge(alpha=1.0))])


df = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["date"])
F = features(df).dropna()
y = df.loc[F.index, "sales"]

# Time series are split CHRONOLOGICALLY, never randomly (random split = leakage).
HOLDOUT = 60
X_tr, y_tr, X_te, y_te = F.iloc[:-HOLDOUT], y.iloc[:-HOLDOUT], F.iloc[-HOLDOUT:], y.iloc[-HOLDOUT:]

model = build().fit(X_tr, y_tr)
naive = df["sales"].shift(7).loc[y_te.index]  # seasonal-naive baseline: same day last week

save(Path(__file__).parent, model, {
    "task": "forecast: daily sales (1-step ahead, last 60 days held out)",
    "test_mae": round(mean_absolute_error(y_te, model.predict(X_te)), 3),
    "baseline_seasonal_naive_mae": round(mean_absolute_error(y_te, naive), 3),
})
