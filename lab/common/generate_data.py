"""Synthetic data for the lab.

Business story: a subscription company wants to predict
  * `churned`          (classification)
  * `next_month_spend` (regression)
and forecast daily `sales` (time series).

Every dataset comes from the same generator; each scenario changes one
knob so each concept has exactly one cause:

  training.csv          baseline world
  production.csv        same world, new customers (healthy production)
  data_drift.csv        covariates moved (P(X) changes), rule unchanged
  concept_drift.csv     the rule changed (P(y|X) changes)
  skewed_serving.csv    same distribution; the serving pipeline differs
                        from training (training-serving skew), see concepts/
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CATEGORICAL = ["plan", "region", "device"]
NUMERIC = ["age", "tenure_months", "monthly_spend", "support_tickets"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "churned"
REG_TARGET = "next_month_spend"


def make_customers(n, seed, *, drift=False, concept_shift=False):
    rng = np.random.default_rng(seed)

    if drift:  # DATA DRIFT: younger, pricier, more mobile, new region mix
        age = rng.normal(28, 7, n).clip(18, 80)
        region_p = [0.1, 0.2, 0.2, 0.5]
        device_p = [0.2, 0.7, 0.1]
        spend_mu = 85
    else:
        age = rng.normal(42, 12, n).clip(18, 80)
        region_p = [0.4, 0.3, 0.2, 0.1]
        device_p = [0.5, 0.4, 0.1]
        spend_mu = 60
    plan = rng.choice(["basic", "pro", "enterprise"], n, p=[0.55, 0.35, 0.10])
    region = rng.choice(["north", "south", "east", "west"], n, p=region_p)
    device = rng.choice(["desktop", "mobile", "tablet"], n, p=device_p)
    tenure = rng.gamma(3, 8, n).clip(1, 120).round()
    plan_bonus = pd.Series(plan).map({"basic": 0, "pro": 40, "enterprise": 150}).values
    spend = (spend_mu + plan_bonus + rng.normal(0, 10, n)).clip(5)
    tickets = rng.poisson(1.5, n)

    # label rule: P(y | X)
    z = (-0.5 + 0.45 * tickets - 0.03 * tenure + 0.015 * (spend - 60)
         + 0.6 * (device == "mobile") - 0.7 * (plan == "enterprise"))
    if concept_shift:
        # CONCEPT DRIFT (e.g. a competitor appears): tickets stop mattering,
        # price sensitivity grows, long-tenure customers now leave.
        z = (-0.5 + 0.01 * tenure + 0.05 * (spend - 60)
             - 0.6 * (device == "mobile") + 0.7 * (plan == "enterprise"))
    churned = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    if concept_shift:
        next_spend = spend * 0.8 + 0.3 * age + rng.normal(0, 8, n)
    else:
        next_spend = (spend * (1 + 0.002 * tenure) - 4 * tickets
                      - 15 * churned + rng.normal(0, 8, n))

    return pd.DataFrame(dict(
        plan=plan, region=region, device=device, age=age.round(1),
        tenure_months=tenure, monthly_spend=spend.round(2),
        support_tickets=tickets, churned=churned,
        next_month_spend=next_spend.clip(0).round(2),
    ))


def make_sales(days, seed, start="2023-01-01"):
    """Daily sales: trend + weekly seasonality + yearly wave + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    weekly = np.array([0, -5, -3, 0, 4, 18, 14])[idx.dayofweek]
    sales = (100 + 0.08 * t + weekly + 8 * np.sin(2 * np.pi * t / 365)
             + rng.normal(0, 3, days))
    return pd.DataFrame({"date": idx, "sales": sales.round(2)})


def main():
    DATA_DIR.mkdir(exist_ok=True)
    make_customers(5000, 1).to_csv(DATA_DIR / "training.csv", index=False)
    make_customers(2000, 2).to_csv(DATA_DIR / "production.csv", index=False)
    make_customers(2000, 3, drift=True).to_csv(DATA_DIR / "data_drift.csv", index=False)
    make_customers(2000, 4, concept_shift=True).to_csv(DATA_DIR / "concept_drift.csv", index=False)
    make_customers(2000, 5).to_csv(DATA_DIR / "skewed_serving.csv", index=False)
    make_sales(730, 6).to_csv(DATA_DIR / "sales.csv", index=False)
    print("wrote", sorted(p.name for p in DATA_DIR.glob("*.csv")))


if __name__ == "__main__":
    main()
