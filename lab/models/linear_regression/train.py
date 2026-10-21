from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from common.utils import REG_TARGET, load, make_pipeline, save, split

# Regression target: how much will the customer spend next month?
X_tr, X_va, X_te, y_tr, y_va, y_te = split(load("training"), target=REG_TARGET)

model = make_pipeline(LinearRegression(), scale=True)
model.fit(X_tr, y_tr)

save(Path(__file__).parent, model, {
    "task": "regression: next_month_spend",
    "train_r2": round(r2_score(y_tr, model.predict(X_tr)), 4),
    "val_r2": round(r2_score(y_va, model.predict(X_va)), 4),
    "test_r2": round(r2_score(y_te, model.predict(X_te)), 4),
    "test_mae": round(mean_absolute_error(y_te, model.predict(X_te)), 3),
})
