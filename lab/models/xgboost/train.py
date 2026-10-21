from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from common.utils import load, make_pipeline, save, split

X_tr, X_va, X_te, y_tr, y_va, y_te = split(load("training"))

model = make_pipeline(XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, eval_metric='logloss', random_state=0), scale=False)
model.fit(X_tr, y_tr)

proba = lambda X: model.predict_proba(X)[:, 1]
save(Path(__file__).parent, model, {
    "task": "classification: churned",
    "train_auc": round(roc_auc_score(y_tr, proba(X_tr)), 4),
    "val_auc": round(roc_auc_score(y_va, proba(X_va)), 4),
    "test_auc": round(roc_auc_score(y_te, proba(X_te)), 4),
    "test_accuracy": round(accuracy_score(y_te, model.predict(X_te)), 4),
})
