"""DATA DRIFT (P(X) changes) vs CONCEPT DRIFT (P(y|X) changes)."""
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import roc_auc_score

from common.utils import CATEGORICAL, FEATURES, NUMERIC, TARGET, load, load_model, make_pipeline, split
from sklearn.ensemble import GradientBoostingClassifier


def drift_report(ref, new):
    """Label-free drift detection: KS test (numeric) / chi-square (categorical), reference vs new."""
    rows = []
    for c in NUMERIC:
        stat = ks_2samp(ref[c], new[c])
        rows.append((c, "KS", stat.statistic, stat.pvalue))
    for c in CATEGORICAL:
        table = pd.concat([ref[c].value_counts(), new[c].value_counts()], axis=1).fillna(0)
        _, p, _, _ = chi2_contingency(table)
        rows.append((c, "chi2", float("nan"), p))
    return pd.DataFrame(rows, columns=["feature", "test", "statistic", "p_value"]).assign(
        drifted=lambda d: d.p_value < 0.01)


def run():
    print("\n=== DATA DRIFT vs CONCEPT DRIFT ===")
    train, prod = load("training"), load("production")
    drifted, concept = load("data_drift"), load("concept_drift")
    model = load_model("gradient_boosting")
    auc = lambda d: roc_auc_score(d[TARGET], model.predict_proba(d[FEATURES])[:, 1])

    for name, d in [("production (healthy)", prod), ("data_drift", drifted), ("concept_drift", concept)]:
        rep = drift_report(train, d)
        print(f"\n{name}: model AUC={auc(d):.3f}; features flagged as drifted: "
              f"{list(rep[rep.drifted].feature) or 'none'}")

    print("\n-> Data drift: inputs shift (detectable WITHOUT labels). The model may degrade or may not:")
    print("   it is only extrapolating into regions it saw little of.")
    print("-> Concept drift: inputs look unchanged (nothing flagged!) yet AUC drops, because the")
    print("   input->label relationship changed. Only labels (delayed ground truth) reveal it.")

    # Remedy: retrain on recent labeled data.
    X_tr, _, X_te, y_tr, _, y_te = split(concept)
    new = make_pipeline(GradientBoostingClassifier(random_state=0)).fit(X_tr, y_tr)
    print(f"   Retrained on concept_drift data -> held-out AUC={roc_auc_score(y_te, new.predict_proba(X_te)[:, 1]):.3f}")


if __name__ == "__main__":
    run()
