"""UNDERFITTING vs OVERFITTING: compare train vs validation score as model capacity grows."""
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from common.utils import load, make_pipeline, split


def run():
    print("\n=== UNDERFITTING / OVERFITTING ===")
    X_tr, X_va, _, y_tr, y_va, _ = split(load("training"))
    X_tr, y_tr = X_tr.iloc[:300], y_tr.iloc[:300]  # small sample makes memorisation easy

    print(f"{'model':28s} {'train AUC':>9s} {'val AUC':>8s}")
    for name, depth in [("underfit (depth=1)", 1), ("balanced (depth=3)", 3), ("overfit (depth=None)", None)]:
        m = make_pipeline(DecisionTreeClassifier(max_depth=depth, random_state=0)).fit(X_tr, y_tr)
        tr = roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])
        va = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
        print(f"{name:28s} {tr:9.3f} {va:8.3f}")
    print("-> Underfit: both scores low (high bias). Overfit: train ~1.0, validation much lower (high variance).")
    print("   Fixes: more capacity / features vs. regularisation, more data, simpler model, early stopping.")


if __name__ == "__main__":
    run()
