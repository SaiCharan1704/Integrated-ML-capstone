<div align="center">

<img src="assets/header.svg" alt="Machine Learning Capstone — Traffic Systems" width="100%" />

<br/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=2800&pause=900&color=38BDF8&center=true&vCenter=true&width=700&lines=Predicting+traffic+volume+with+10+regression+models;Classifying+accident+severity+across+3+injury+levels;Built+with+scikit-learn%2C+pandas+and+a+lot+of+GridSearchCV" alt="Typing SVG" />
</a>

<br/><br/>

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
<img src="https://img.shields.io/badge/pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
<img src="https://img.shields.io/badge/Status-Active-22C55E?style=for-the-badge" />

</div>

---

## Overview

An end-to-end machine learning capstone built around a single theme: **road traffic systems**. Three tracks share one consistent methodology — audit, clean, engineer, pipeline, model, tune, interpret.

| Track | Dataset | Rows × Cols | Task |
|:--|:--|:--:|:--|
| Regression | Metro Interstate Traffic Volume | 48,204 × 9 | Predict hourly `traffic_volume` |
| Classification | Road Traffic Accident Severity | 12,316 × 32 | Predict 3-class injury severity |
| Clustering | Leeds Road Accidents | — | Unsupervised structure discovery |

---

## Results

<div align="center">
  <img src="assets/results.svg" alt="Animated model comparison" width="100%" />
</div>

<div align="center">

| | Best model | Headline metric | Runner-up |
|:--|:--|:--:|:--|
| **Regression** | Random Forest *(tuned)* | **R² 0.9561** · RMSE 425.8 · MAE 238.6 | Decision Tree (0.9417) |
| **Classification** | K-Nearest Neighbors | **83.85% acc** · weighted F1 0.7893 | Decision Tree (0.7633) |

</div>

---

## Deep dive

<details>
<summary><b>📈 Regression — Metro Interstate Traffic Volume</b></summary>

<br/>

**Pipeline**

- Duplicates dropped; anomalous `temp == 0` and a single 9831.30 mm rainfall reading treated as missing
- Missing `holiday` filled with `"None"`; median imputation for numeric, most-frequent for categorical
- `date_time` expanded into `hour`, `day_of_week`, `month`, `year`, `is_weekend`
- Engineered `traffic_period` → Morning / Afternoon / Evening / Night
- `StandardScaler` + `OneHotEncoder` inside a `ColumnTransformer`, fit on train only
- 80 / 20 split, `random_state=42`

**Ten algorithms compared**

`Linear` · `Ridge` · `Lasso` · `ElasticNet` · `Polynomial (deg 2)` · `Decision Tree` · `Random Forest` · `Gradient Boosting` · `SVR (RBF)` · `KNN`

**Tuning** — `GridSearchCV`, 5-fold

| Model | Best params | CV R² |
|:--|:--|:--:|
| Random Forest | `n_estimators=200, max_depth=20, min_samples_split=5` | 0.9500 |
| Gradient Boosting | `n_estimators=200, learning_rate=0.1, max_depth=4` | 0.9468 |

**Feature importance (tuned RF)**

```
traffic_period_Night  ████████████████████████████  0.5838
hour                  ████████████                  0.2540
day_of_week           ███                           0.0633
is_weekend            ██                            0.0376
```

Time dominates weather — traffic is a clock, not a barometer.

</details>

<details>
<summary><b>🚦 Classification — Road Traffic Accident Severity</b></summary>

<br/>

**The imbalance problem**

```
Slight Injury   ██████████████████████████████████  10,415
Serious Injury  █████                                1,743
Fatal injury    ▌                                      158
```

Accuracy alone is misleading here, so **weighted F1** is reported alongside it, and `class_weight="balanced"` is used where the estimator supports it.

**Pipeline**

- String `"na"` converted to real `NaN`, duplicates removed
- `Time` → `Hour` → `Traffic_Period` (Afternoon peaks at 3,897 accidents; Night lowest at 1,611)
- Leakage columns (`Casualty_severity`) dropped before modelling
- Stratified 80 / 20 split to preserve class proportions

**Five algorithms compared**

| Algorithm | Accuracy | Weighted F1 |
|:--|:--:|:--:|
| K-Nearest Neighbors | 0.8385 | 0.7893 |
| Decision Tree | 0.7593 | 0.7633 |
| Support Vector Machine | 0.6924 | 0.7267 |
| Logistic Regression | 0.5069 | 0.5905 |
| Gaussian Naive Bayes | 0.0913 | 0.1241 |

Naive Bayes collapses — its conditional-independence assumption does not survive 30+ one-hot encoded accident attributes.

**Interpretation** — per-class logistic coefficients and a depth-3 decision tree plot.

</details>

<details>
<summary><b>🧩 Clustering — Leeds Road Accidents</b></summary>

<br/>

Unsupervised track: K-Means and hierarchical clustering with PCA / t-SNE projections for visual validation.

*In progress.*

</details>

---

## Repository structure

```
.
├── data/
│   ├── Metro_Interstate_Traffic_Volume.csv
│   ├── RTA Dataset.csv
│   └── leeds_accidents.csv
├── notebooks/
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
├── assets/
│   ├── header.svg
│   └── results.svg
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/SaiCharan1704/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter lab notebooks/
```

<details>
<summary><b>requirements.txt</b></summary>

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyterlab
```

</details>

---

## Methodology notes

- **No leakage.** Every transformer is fit on the training split only, then applied to test.
- **One split, many models.** All algorithms in a track see the identical split and preprocessing, so the comparison is fair.
- **Metrics matched to the problem.** R² / RMSE / MAE for regression; accuracy *and* weighted F1 for the imbalanced classification target.
- **Tuning is cross-validated.** `GridSearchCV` with 5 folds; the held-out test set is touched only once, at the end.

---

<div align="center">

<img src="https://img.shields.io/badge/Course-23CSE301%20Machine%20Learning-0EA5E9?style=flat-square" />
<img src="https://img.shields.io/badge/Amrita%20Vishwa%20Vidyapeetham-B91C1C?style=flat-square" />

<br/><br/>

**Sai** · [@SaiCharan1704](https://github.com/SaiCharan1704)

<img src="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake-dark.svg" alt="" width="0" height="0" />

<sub>⭐ Star the repo if the traffic analysis was useful.</sub>

</div>
