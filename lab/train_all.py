"""Train every model. Each one is saved in models/<name>/ (model.joblib + metrics.json)."""
import runpy
from pathlib import Path

for d in sorted((Path(__file__).parent / "models").iterdir()):
    if (d / "train.py").exists():
        runpy.run_path(str(d / "train.py"), run_name="__main__")
