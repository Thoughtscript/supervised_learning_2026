from pathlib import Path

from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from common.utils import load, make_pipeline, save, split

X_tr, X_va, X_te, y_tr, y_va, y_te = split(load("training"))

# Soft voting: average the probabilities of diverse models. Errors that are
# not shared between models cancel out. The shared one-hot pipeline is fit once.
voter = VotingClassifier([
    ("logreg", LogisticRegression(max_iter=1000)),
    ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=10, random_state=0, n_jobs=-1)),
    ("gb", GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=0)),
    ("xgb", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, eval_metric="logloss", random_state=0)),
], voting="soft")

model = make_pipeline(voter, scale=True)
model.fit(X_tr, y_tr)

proba = lambda X: model.predict_proba(X)[:, 1]
save(Path(__file__).parent, model, {
    "task": "classification: churned (soft-voting ensemble)",
    "train_auc": round(roc_auc_score(y_tr, proba(X_tr)), 4),
    "val_auc": round(roc_auc_score(y_va, proba(X_va)), 4),
    "test_auc": round(roc_auc_score(y_te, proba(X_te)), 4),
    "test_accuracy": round(accuracy_score(y_te, model.predict(X_te)), 4),
})
