from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from common.utils import load, make_pipeline, save, split

X_tr, X_va, X_te, y_tr, y_va, y_te = split(load("training"))

model = make_pipeline(RandomForestClassifier(n_estimators=200, min_samples_leaf=10, random_state=0, n_jobs=-1), scale=False)
model.fit(X_tr, y_tr)

proba = lambda X: model.predict_proba(X)[:, 1]
save(Path(__file__).parent, model, {
    "task": "classification: churned",
    "train_auc": round(roc_auc_score(y_tr, proba(X_tr)), 4),
    "val_auc": round(roc_auc_score(y_va, proba(X_va)), 4),
    "test_auc": round(roc_auc_score(y_te, proba(X_te)), 4),
    "test_accuracy": round(accuracy_score(y_te, model.predict(X_te)), 4),
})
