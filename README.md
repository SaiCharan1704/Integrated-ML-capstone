# Machine Learning Capstone — Traffic Systems

> Three supervised and unsupervised tracks on one theme: how road traffic behaves, and what predicts it.
> 15 algorithms, one leak-free pipeline per track, cross-validated tuning, honest metrics.

`Python 3.10+` · `scikit-learn` · `pandas` · `Jupyter` · 23CSE301 Machine Learning · Amrita Vishwa Vidyapeetham

---

## Contents

- [Overview](#overview)
- [Headline results](#headline-results)
- [Regression — traffic volume](#regression--metro-interstate-traffic-volume)
- [Classification — accident severity](#classification--road-traffic-accident-severity)
- [Clustering — Leeds accidents](#clustering--leeds-road-accidents)
- [Repository structure](#repository-structure)
- [Quickstart](#quickstart)
- [Methodology notes](#methodology-notes)

---

## Overview

| Track | Dataset | Shape | Task |
|:--|:--|:--:|:--|
| Regression | Metro Interstate Traffic Volume | 48,204 × 9 | Predict hourly `traffic_volume` |
| Classification | Road Traffic Accident Severity | 12,316 × 32 | Predict 3-class injury severity |
| Clustering | Leeds Road Accidents | — | Discover unsupervised structure |

Every track follows the same spine: **audit → clean → engineer → pipeline → model → tune → interpret.** Preprocessing is fit on the training split only, and every algorithm within a track sees the identical split, so the comparisons are fair rather than flattering.

---

## Headline results

| Track | Best model | Headline metric | Runner-up |
|:--|:--|:--|:--|
| Regression | Random Forest *(tuned)* | **R² 0.9561** · RMSE 425.82 · MAE 238.57 | Decision Tree — R² 0.9417 |
| Classification | K-Nearest Neighbors | **83.85% accuracy** · weighted F1 0.7893 | Decision Tree — F1 0.7633 |

**One-line takeaway:** traffic volume is driven by the clock, not the weather — `traffic_period_Night` and `hour` together account for ~84% of the tuned Random Forest's feature importance, while temperature, rain and cloud cover correlate with the target at 0.13 or below.

---

## Regression — Metro Interstate Traffic Volume

**Goal.** Predict hourly traffic volume from weather, holiday and time information.

### Model comparison — test set

| Rank | Model | R² | RMSE | MAE |
|:--:|:--|:--:|:--:|:--:|
| 1 | Random Forest | 0.9546 | 425.82 | 238.57 |
| 2 | Decision Tree | 0.9417 | 482.93 | 283.21 |
| 3 | Gradient Boosting | 0.9313 | 524.20 | 343.25 |
| 4 | Polynomial (deg 2) | 0.8510 | 771.85 | 575.59 |
| 5 | KNN | 0.8307 | 822.72 | 579.25 |
| 6 | SVR (RBF) | 0.8294 | 825.96 | 601.18 |
| 7 | Lasso | 0.7314 | 1036.34 | 795.32 |
| 8 | Ridge | 0.7312 | 1036.65 | 795.55 |
| 9 | Linear | 0.7311 | 1036.80 | 795.62 |
| 10 | ElasticNet | 0.7169 | 1063.81 | 842.84 |

The ~0.22 R² gap between the tree ensembles and the linear family is the story: the relationship between time of day and traffic volume is sharply non-linear, and no amount of regularisation rescues a straight line.

<details>
<summary><b>Cleaning and feature engineering</b></summary>

<br/>

- Duplicate rows removed before any modelling.
- `temp == 0` (Kelvin) treated as missing — physically impossible and far outside the 242–321 K IQR band.
- A single 9,831.30 mm/hour rainfall reading treated as missing; the next-highest value is orders of magnitude smaller.
- Missing `holiday` filled with `"None"`, which is the semantically correct value rather than an imputation.
- Genuine extreme values were **not** deleted — only demonstrable data errors were.

`date_time` was expanded into:

| Feature | Meaning |
|:--|:--|
| `hour` | 0–23 |
| `day_of_week` | Monday = 0 |
| `month`, `year` | seasonal and long-run drift |
| `is_weekend` | binary |
| `traffic_period` | Morning (5–11), Afternoon (12–16), Evening (17–20), Night (21–4) |

Justification: traffic volume follows recurring commute cycles, so the raw timestamp carries far more signal once decomposed than it does as a single opaque column.

</details>

<details>
<summary><b>Preprocessing pipeline</b></summary>

<br/>

```
ColumnTransformer
├── numerical   → SimpleImputer(median)        → StandardScaler
└── categorical → SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
```

80 / 20 split, `random_state=42`, fit on train only and applied to test. The same fitted preprocessor feeds all ten algorithms.

</details>

<details>
<summary><b>Hyperparameter tuning — GridSearchCV, 5-fold</b></summary>

<br/>

| Model | Best parameters | Best CV R² |
|:--|:--|:--:|
| Random Forest | `n_estimators=200, max_depth=20, min_samples_split=5` | 0.9500 |
| Gradient Boosting | `n_estimators=200, learning_rate=0.1, max_depth=4` | 0.9468 |

Five-fold cross-validation on the untuned top two gave Random Forest 0.9483 ± 0.0038 and Decision Tree 0.9344 ± 0.0040 — low variance across folds, so the test-set ranking is stable rather than a lucky split. The tuned Random Forest reached **R² 0.9561** on the held-out test set.

</details>

<details>
<summary><b>Feature importance — tuned Random Forest</b></summary>

<br/>

```
traffic_period_Night   ############################   0.5838
hour                   ############                   0.2540
day_of_week            ###                            0.0633
is_weekend             ##                             0.0376
temp                   #                              (minor)
```

Time features dominate. Weather variables contribute, but marginally.

</details>

<details>
<summary><b>Diagnostics</b></summary>

<br/>

- **Predicted vs actual:** points cluster tightly along the diagonal, consistent with R² 0.9561.
- **Residuals:** centred on zero — no strong systematic bias — but the spread widens at higher predicted volumes, so the model is less precise during peak traffic than during quiet hours.

</details>

---

## Classification — Road Traffic Accident Severity

**Goal.** Predict injury severity from 32 accident attributes.

### The imbalance problem

| Class | Count | Share |
|:--|--:|--:|
| Slight Injury | 10,415 | 84.6% |
| Serious Injury | 1,743 | 14.2% |
| Fatal injury | 158 | 1.3% |

A model that predicts "Slight Injury" for everything scores 84.6% accuracy while being useless. **Weighted F1 is therefore reported alongside accuracy**, and `class_weight="balanced"` is used wherever the estimator supports it.

### Model comparison — stratified test set

| Rank | Algorithm | Accuracy | Weighted F1 |
|:--:|:--|:--:|:--:|
| 1 | K-Nearest Neighbors | 0.8385 | **0.7893** |
| 2 | Decision Tree | 0.7593 | 0.7633 |
| 3 | Support Vector Machine | 0.6924 | 0.7267 |
| 4 | Logistic Regression | 0.5069 | 0.5905 |
| 5 | Gaussian Naive Bayes | 0.0913 | 0.1241 |

Gaussian Naive Bayes collapses because its two core assumptions both fail here: the features are one-hot encoded categoricals rather than Gaussian, and 30+ accident attributes are plainly not conditionally independent. Logistic Regression's low accuracy is the cost of `class_weight="balanced"` — it trades majority-class accuracy for minority-class recall, which is arguably the right trade for fatal-injury prediction.

<details>
<summary><b>Cleaning and feature engineering</b></summary>

<br/>

- The literal string `"na"` converted to proper `NaN` so imputers could see it.
- Duplicates removed; rows with a missing target dropped.
- `Casualty_severity` dropped as a leakage column — it encodes the outcome.
- `Time` parsed → `Hour` → `Traffic_Period`.

Accidents by traffic period: Afternoon 3,897 (highest), Night 1,611 (lowest). Accident occurrence clearly varies by time of day, so the engineered period carries signal.

</details>

<details>
<summary><b>Preprocessing and evaluation</b></summary>

<br/>

Same `ColumnTransformer` structure as the regression track — median / most-frequent imputation, `StandardScaler`, `OneHotEncoder(handle_unknown="ignore")` — with a **stratified** 80 / 20 split so class proportions survive the split.

Each model was evaluated with a confusion matrix and a full per-class classification report, not just the summary table above. Interpretation came from per-class Logistic Regression coefficients (one row per severity class, sign indicating direction of contribution) and a depth-3 Decision Tree plot.

</details>

---

## Clustering — Leeds Road Accidents

Unsupervised track: K-Means and hierarchical clustering, with PCA and t-SNE projections used to sanity-check whether the discovered clusters are genuinely separated or an artefact of the distance metric.

*In progress.*

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

**requirements.txt**

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyterlab
```

Notebooks read from `../data/`, so run them from inside `notebooks/`.

---

## Methodology notes

**No leakage.** Every transformer is fit on the training split alone, then applied to test. Outcome-derived columns are dropped before modelling.

**One split, many models.** All algorithms in a track share the same split and the same fitted preprocessor, so differences in score reflect the algorithm and nothing else.

**Metrics matched to the problem.** R² / RMSE / MAE for regression; accuracy *and* weighted F1 for the imbalanced classification target, because accuracy alone would reward a model that never predicts a fatality.

**Tuning is cross-validated.** `GridSearchCV` with 5 folds selects hyperparameters; the held-out test set is touched once, at the end, to report the final number.

**Errors were fixed, outliers were not.** Impossible values (0 K temperature, a 9,831 mm rainfall reading) were treated as missing. Genuine extremes — accidents with many vehicles or casualties — were kept, because they are the observations that matter most.

---

**Sai** · [@SaiCharan1704](https://github.com/SaiCharan1704)
