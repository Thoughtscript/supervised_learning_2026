# ML Lab: models, one-hot encoding, and what goes wrong in production

Only Docker is needed on the host. Everything (Python, scikit-learn, XGBoost) runs in one container.

```bash
cd lab
docker compose up --build        # data -> train -> concepts
docker compose run --rm shell    # optional shell (profile "tools")
```

The `lab/` folder is mounted, so generated CSVs and saved models appear on your host.

> Note: these are scikit-learn / XGBoost models, not TensorFlow. Random forests, gradient boosting, XGBoost,
> and linear/logistic regression are not TensorFlow models, so TensorFlow would add a ~500 MB dependency for no benefit.

## Layout

| Path | Purpose |
|---|---|
| `common/generate_data.py` | Synthetic customers (churn + next-month spend) and daily sales. One generator, one knob per scenario |
| `common/utils.py` | 60/20/20 split, one-hot `preprocessor`, `make_pipeline`, save/load |
| `models/<name>/train.py` | `xgboost`, `random_forest`, `gradient_boosting`, `linear_regression`, `logistic_regression`, `ensemble_learning`, `time_series_forecasting`. Each writes `model.joblib` + `metrics.json` into its own folder |
| `concepts/*.py` | One script per concept; `run_all.py` runs them |
| `data/` | Generated CSVs (git-ignored) |

Run a single step: `docker compose run --rm train python models/xgboost/train.py`

## Datasets

| File | Meaning |
|---|---|
| `training.csv` | Baseline; used to build every model |
| `production.csv` | Healthy production: same world, new customers |
| `data_drift.csv` | Inputs changed (younger, pricier, mobile, new region mix); the rule is the same |
| `concept_drift.csv` | Inputs like training, but the input→churn rule changed |
| `skewed_serving.csv` | Healthy data, served through a buggy feature pipeline |
| `sales.csv` | 2 years of daily sales for forecasting |

## One-hot encoding

`plan`, `region`, `device` (and day-of-week in the forecaster) are one-hot encoded inside the saved `Pipeline`,
with `handle_unknown="ignore"`. Numeric columns are scaled only for the linear models. Trees don't need it.

## Concepts and what you'll see

| Concept | Demo | Signature |
|---|---|---|
| Data leakage | `leakage.py` | Feature created *after* the label: offline AUC ≈ 0.98, production ≈ 0.67 |
| Underfitting | `fit.py` | depth-1 tree: low train *and* validation AUC |
| Overfitting | `fit.py` | unbounded tree: train AUC 1.0, validation far lower |
| Data drift | `drift.py` | KS / chi-square flag age, spend, region, device without needing labels |
| Concept drift | `drift.py` | No feature flagged, yet AUC drops; retraining on new labels recovers it |
| Training-serving skew | `skew.py` | Same data, different feature code (years vs months, `PRO` vs `pro`): silent AUC loss |

Train/validation/test splitting, leakage-safe pipelines and the chronological split for time series are in
`common/utils.py` and `models/time_series_forecasting/train.py`.

## Ideas to extend

- Tune hyperparameters on the validation set and touch the test set once.
- Add a rolling AUC monitor over `production.csv` chunks.
- Add `scikit-learn`'s `permutation_importance` to catch leaky features.
