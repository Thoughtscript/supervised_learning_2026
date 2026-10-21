"""DATA LEAKAGE: the model sees information at training time that will not exist at prediction time."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from common.generate_data import CATEGORICAL, NUMERIC
from common.utils import TARGET, load, preprocessor, split
from sklearn.pipeline import Pipeline


def fit_eval(df_train_src, extra=None):
    num = NUMERIC + (extra or [])
    X_tr, _, X_te, y_tr, _, y_te = split(df_train_src)
    X_tr, X_te = df_train_src.loc[X_tr.index, CATEGORICAL + num], df_train_src.loc[X_te.index, CATEGORICAL + num]
    m = Pipeline([("prep", preprocessor(categorical=CATEGORICAL, numeric=num)),
                  ("model", GradientBoostingClassifier(random_state=0))]).fit(X_tr, y_tr)
    return m, num


def run():
    print("\n=== DATA LEAKAGE ===")
    train, prod = load("training"), load("production")

    # `retention_offer_sent` is recorded AFTER a customer cancels (the winback
    # team emails churners). It is a consequence of the label, not a cause.
    train["retention_offer_sent"] = (train[TARGET] & (train.index % 20 != 0)).astype(int)
    # In production we predict BEFORE anyone churns, so the column is always 0.
    prod["retention_offer_sent"] = 0

    for label, extra in [("clean features", None), ("with leaky feature", ["retention_offer_sent"])]:
        m, num = fit_eval(train, extra)
        _, _, X_te, _, _, y_te = split(train)
        offline = roc_auc_score(y_te, m.predict_proba(train.loc[X_te.index, CATEGORICAL + num])[:, 1])
        live = roc_auc_score(prod[TARGET], m.predict_proba(prod[CATEGORICAL + num])[:, 1])
        print(f"{label:20s} offline test AUC={offline:.3f}   production AUC={live:.3f}")
    print("-> The leaky model looks near-perfect offline and collapses in production.")
    print("   Red flag: suspiciously high scores. Fix: only use features available at prediction time,")
    print("   split BEFORE fitting scalers/encoders (see common/utils.py), split time series chronologically.")


if __name__ == "__main__":
    run()
