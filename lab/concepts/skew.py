"""TRAINING-SERVING SKEW: same data distribution, but serving computes features differently than training."""
from sklearn.metrics import roc_auc_score

from common.utils import FEATURES, TARGET, load, load_model


def buggy_serving_pipeline(df):
    """Imagine the online service was reimplemented by another team."""
    df = df.copy()
    df["tenure_months"] = df["tenure_months"] / 12   # sends YEARS instead of months
    df["plan"] = df["plan"].str.upper()               # 'pro' -> 'PRO' (category unseen in training)
    return df


def run():
    print("\n=== TRAINING-SERVING SKEW ===")
    model = load_model("gradient_boosting")
    d = load("skewed_serving")
    auc = lambda X: roc_auc_score(d[TARGET], model.predict_proba(X[FEATURES])[:, 1])
    print(f"same data, training-consistent features: AUC={auc(d):.3f}")
    print(f"same data, buggy serving transform     : AUC={auc(buggy_serving_pipeline(d)):.3f}")
    print("-> The distributions of the raw data are identical; only the feature code differs.")
    print("   Because OneHotEncoder(handle_unknown='ignore') does not crash, the bug is silent.")
    print("   Fixes: ship preprocessing inside the saved Pipeline, log serving features, compare to training.")


if __name__ == "__main__":
    run()
