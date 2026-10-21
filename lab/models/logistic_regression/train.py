from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

from common.utils import load, make_pipeline, save, split

X_tr, X_va, X_te, y_tr, y_va, y_te = split(load("training"))

model = make_pipeline(LogisticRegression(max_iter=1000), scale=True)
model.fit(X_tr, y_tr)

proba = lambda X: model.predict_proba(X)[:, 1]
save(Path(__file__).parent, model, {
    "task": "classification: churned",
    "train_auc": round(roc_auc_score(y_tr, proba(X_tr)), 4),
    "val_auc": round(roc_auc_score(y_va, proba(X_va)), 4),
    "test_auc": round(roc_auc_score(y_te, proba(X_te)), 4),
    "test_accuracy": round(accuracy_score(y_te, model.predict(X_te)), 4),
})
