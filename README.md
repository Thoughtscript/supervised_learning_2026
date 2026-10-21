# supervised_learning_2026

[![agents-cli 1.11.0](https://img.shields.io/badge/agents--cli-1.11.0-blue.svg)](https://google.github.io/agents-cli/guide/getting-started/)
[![TensorFlow 2.20.0](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00)](https://github.com/tensorflow/tensorflow/releases/tag/v2.20.0)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB)](https://www.python.org/downloads/)
[![scikit-learn 1.9.1](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E)](https://github.com/scikit-learn/scikit-learn/releases/tag/1.9.1)
[![XGBoost 3.2.0](https://img.shields.io/badge/XGBoost-3.2.0-189FDD)](https://github.com/dmlc/xgboost/releases/tag/v3.2.0)
[![pandas 3.0.6](https://img.shields.io/badge/pandas-3.0.6-150458)](https://pandas.pydata.org/)
[![NumPy 2.4.6](https://img.shields.io/badge/NumPy-2.4.6-013243)](https://numpy.org/)
[![SciPy 1.17.1](https://img.shields.io/badge/SciPy-1.17.1-8CAAE6)](https://scipy.org/)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED)](https://docs.docker.com/compose/)

> *A simple **Supervised Learning** Lab.*

## Overview

I asked GitHub Copilot to quickly scaffold an example using these **Supervised Learning** (labeled data) techniques:
* **XGBoost**
* **Random Forest**
* **Gradient Boosting**
* **Linear Regression**
* **Logistic Regression**
* **Ensemble Learning**
* **Time-Series Forecasting**

Start with initial data (randomized, CSV format) then demonstrate the following scenarios:
* **Data Leakage**
* **Underfitting**
* **Overfitting**
* **Data Drift**
* **Concept Drift**
* **Skew**

The above frequently show up in the GCP PMLE but there aren't that many hands-on labs for these in [skills.google](https://www.skills.google/).

GitHub Copilot diligently implemented this (very literally) and created a Docker example in sequential stages: generate random Data, train the specified Models, then demonstrate scenarios like Data Drift.

## Use

```bash
docker compose up
```

> Note that Data, Models, Metrics will be recreated within the Container. I've included generated examples in the source for study purposes.

## Resources and Links

1. https://render-static-fs.onrender.com/study_guide_notes_2023.gcp.html#gcp-pmle-working-notes
1. https://www.skills.google/