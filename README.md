<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg" width="80" alt="NASA Logo"/>

# 🪨 NASA Asteroid Hazard Classification

**An end-to-end machine learning pipeline for binary classification of Near-Earth Objects (NEOs) as Potentially Hazardous or Safe — powered by NASA's NeoWs dataset and gradient-boosted ensemble methods.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red?logo=python&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-brightgreen)](https://lightgbm.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-green?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

> _"A major asteroid impact is statistically inevitable on the scale of millennia. Early detection and accurate classification is the cornerstone of planetary defense."_

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Business Context & Motivation](#-business-context--motivation)
- [Dataset Description](#-dataset-description)
- [Project Structure](#-project-structure)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering & Preprocessing](#-feature-engineering--preprocessing)
- [Model Development & Comparison](#-model-development--comparison)
- [Results Summary](#-results-summary)
- [Key Insights](#-key-insights)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Future Work](#-future-work)
- [References](#-references)

---

## 🌌 Project Overview

This project builds a production-ready **binary classification system** to determine whether a Near-Earth Object (NEO) is **Potentially Hazardous (PHA)** or **Non-Hazardous**, based on its physical and orbital characteristics sourced from NASA's NeoWs API.

Two industry-standard gradient boosting frameworks — **XGBoost** and **LightGBM** — are trained, evaluated, and compared across both default and tuned hyperparameter configurations, with a focus on **maximising recall** for the hazardous class to minimise the risk of missing a genuinely dangerous asteroid.

---

## 🎯 Business Context & Motivation

| Stakeholder | Use Case |
|---|---|
| **NASA / CNEOS** | Automated PHA flagging for follow-up orbital study |
| **ESA Planetary Defense Office** | Risk prioritisation for deflection mission planning |
| **NEOWISE / Space Telescopes** | Pre-screening candidates before costly deep observation |
| **Research Institutions** | Building early-warning systems and impact risk models |

> **Why prioritise recall over precision?**  
> A **false negative** (missing a hazardous asteroid) is a catastrophic, irreversible outcome. A **false positive** (over-flagging a safe asteroid) merely costs monitoring resources. This asymmetric cost justifies optimising for recall on the hazardous class, even at a modest precision trade-off.

---

## 📊 Dataset Description

**Source:** [NASA NeoWs — Kaggle](https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification)  
**Collected by:** NASA's Near-Earth Object Web Service (NeoWs) — JPL Small Body Database  
**Size:** 4,687 rows × 40 columns (raw)

### Target Variable

| Label | Class | Count | Share |
|---|---|:---:|:---:|
| `True` → `1` | Potentially Hazardous Asteroid (PHA) | ~1,673 | ~35.7% |
| `False` → `0` | Non-Hazardous | ~3,014 | ~64.3% |

> Moderate class imbalance (~64/36). Addressed via recall-focused evaluation metrics.

### Feature Overview (Raw Dataset)

| Feature | Description | Unit |
|---|---|---|
| `Absolute Magnitude` | Intrinsic brightness (lower = brighter/larger) | — |
| `Est Dia in KM (min/max)` | Estimated diameter range | km |
| `Relative Velocity km per sec` | Speed relative to Earth at close approach | km/s |
| `Miss Dist.(Astronomical)` | Closest approach distance | AU |
| `Orbit Uncertainity` | Uncertainty index (0 = precise, 9 = unreliable) | ordinal |
| `Minimum Orbit Intersection` | MOID — minimum distance between orbital paths | AU |
| `Jupiter Tisserand Invariant` | Orbital energy metric relative to Jupiter | — |
| `Eccentricity` | Deviation from a circular orbit (0 = circle, 1 = parabola) | — |
| `Semi Major Axis` | Half the longest orbital axis | AU |
| `Inclination` | Angle of orbit relative to ecliptic plane | degrees |
| `Orbital Period` | Time for one full revolution | days |
| `Perihelion Distance` | Closest distance to the Sun | AU |
| `Aphelion Dist` | Farthest distance from the Sun | AU |
| `Mean Motion` | Average angular speed | °/day |

---

## 📁 Project Structure

```
NASA-Asteroid-Classification/
│
├── input_data/
│   └── nasa.csv                            # Raw NASA NeoWs dataset
│
├── notebook/
│   └── NASA_Asteroid_Classification.ipynb  # Full EDA + modelling pipeline
│
└── README.md
```

---

## 🔍 Exploratory Data Analysis

A thorough EDA was conducted prior to modelling to understand data quality, feature distributions, class separation, and multicollinearity.

### 1. Data Quality Audit

| Check | Finding | Action |
|---|---|---|
| Null values | ✅ None found | — |
| Duplicate rows | ✅ None found | — |
| Identifier columns | `Neo Reference ID`, `Name`, `Orbit ID` | Dropped |
| Date columns | `Close Approach Date`, `Epoch Date Close Approach`, `Orbit Determination Date` | Dropped |
| Single-value columns | `Orbiting Body` (100% "Earth"), `Equinox` (100% "J2000") | Dropped — zero variance |
| Redundant unit columns | Diameter in M, Miles, Feet; velocity in km/hr & mph; miss dist in lunar, km, miles | Dropped — kept one per physical quantity |
| **Final features** | **19 predictive features** after cleaning | — |

### 2. Class Distribution

The `Hazardous` target is encoded via `LabelEncoder` (`False → 0`, `True → 1`). The ~64/36 split represents moderate imbalance — sufficient to require recall-aware evaluation but not extreme enough to require synthetic oversampling.

### 3. Key Univariate Findings

| Feature | Hazardous (PHA) | Non-Hazardous | Interpretation |
|---|:---:|:---:|---|
| `Absolute Magnitude` (mean) | ~20.5 | ~23.5 | PHAs are intrinsically brighter → physically larger |
| `Est Dia in KM(min)` (mean) | ~0.56 km | ~0.32 km | PHAs are ~75% wider on average |
| `Minimum Orbit Intersection` (median) | ~0.020 AU | ~0.095 AU | PHAs orbit much closer to Earth's path |
| `Miss Dist.(Astronomical)` (median) | ~0.021 AU | ~0.198 AU | PHAs approach Earth significantly more closely |
| `Relative Velocity km per sec` (mean) | ~18.2 | ~14.1 | PHAs are modestly faster on close approach |

### 4. Correlation Analysis & Multicollinearity

Two correlation heatmaps were generated — before and after feature removal — to diagnose and resolve collinearity:

**High-correlation groups identified and resolved:**

| Group | Correlated Features | Resolution |
|---|---|---|
| Diameter | `Est Dia in KM/M/Miles/Feet (min & max)` — 8 columns | Kept `Est Dia in KM(min)` only |
| Velocity | `Relative Velocity km per sec`, `km per hr`, `Miles per hour` | Kept `km per sec` only |
| Miss Distance | `Miss Dist.(Astronomical)`, `(kilometers)`, `(miles)`, `(lunar)` | Kept `(Astronomical)` only |

> After cleaning, the post-heatmap confirmed no remaining high-collinearity pairs among the 19 retained features.

### 5. Notable EDA Observations

- `Minimum Orbit Intersection` (MOID) exhibits the sharpest class separation of all features — near-zero MOID is almost exclusively associated with PHAs
- `Absolute Magnitude` and `Est Dia in KM(min)` are inversely related and jointly confirm that PHAs skew larger and brighter
- `Orbit Uncertainity` tends to be lower for PHAs — frequently-approaching objects receive more observational coverage, reducing orbital uncertainty
- Right-skew is present in velocity, diameter, and miss distance — typical for astronomical datasets with a wide dynamic range

---

## ⚙️ Feature Engineering & Preprocessing

| Step | Detail |
|---|---|
| **Label encoding** | `Hazardous` encoded with `LabelEncoder` (`False=0`, `True=1`) |
| **Column removal** | IDs, dates, zero-variance columns dropped (see EDA) |
| **Redundant unit removal** | All duplicate-unit representations collapsed to one column per physical quantity |
| **Train/test split** | 70/30, `random_state=101` |
| **Feature scaling** | Not applied — gradient boosting is scale-invariant by design |

**Final 19 features used for modelling:**

```
Absolute Magnitude, Est Dia in KM(min), Relative Velocity km per sec,
Miss Dist.(Astronomical), Orbit Uncertainity, Minimum Orbit Intersection,
Jupiter Tisserand Invariant, Epoch Osculation, Eccentricity, Semi Major Axis,
Inclination, Asc Node Longitude, Orbital Period, Perihelion Distance,
Perihelion Arg, Aphelion Dist, Perihelion Time, Mean Anomaly, Mean Motion
```

---

## 🤖 Model Development & Comparison

Two gradient boosting frameworks were trained and evaluated across two configurations each: **default hyperparameters** and **tuned hyperparameters**.

### Models Trained

| # | Model | Configuration | Key Parameters |
|---|---|---|---|
| 1 | XGBoost | Default | Library defaults |
| 2 | LightGBM | Default | Library defaults |
| 3 | XGBoost | Tuned | `n_estimators=100`, `learning_rate=0.1`, `eval_metric='logloss'`, `random_state=101` |
| 4 | LightGBM | Tuned | `n_estimators=100`, `learning_rate=0.1`, `random_state=101` |

---

### 📈 Performance Comparison — All Configurations

> Evaluated on the 30% held-out test set (`test_size=0.3`, `random_state=101`).  
> Precision, Recall, and F1 are reported for the **Hazardous (positive) class**.

| Model | Config | Accuracy | Precision (PHA) | Recall (PHA) | F1-Score (PHA) |
|---|---|:---:|:---:|:---:|:---:|
| XGBoost | Default | **99.93%** | 1.000 | 0.998 | 0.999 |
| LightGBM | Default | 99.86% | 0.998 | 0.998 | 0.998 |
| XGBoost | Tuned | **99.93%** | 1.000 | 0.998 | 0.999 |
| LightGBM | Tuned | 99.86% | 0.998 | 0.998 | 0.998 |

> Both models achieve near-perfect classification. This reflects the high separability of the NASA NeoWs dataset — `Minimum Orbit Intersection` and `Absolute Magnitude` together provide an almost clean decision boundary that gradient boosting exploits fully.

---

### 🔢 Confusion Matrices

**XGBoost — Default & Tuned**

```
                      Predicted: Safe    Predicted: Hazardous
Actual: Safe               904                    0
Actual: Hazardous            1                  502
```

| Metric | Value |
|---|---|
| True Positives (PHAs correctly flagged) | 502 |
| False Negatives (PHAs missed) | **1** |
| False Positives (safe over-flagged) | 0 |
| True Negatives | 904 |

---

**LightGBM — Default & Tuned**

```
                      Predicted: Safe    Predicted: Hazardous
Actual: Safe               903                    1
Actual: Hazardous            1                  502
```

| Metric | Value |
|---|---|
| True Positives (PHAs correctly flagged) | 502 |
| False Negatives (PHAs missed) | **1** |
| False Positives (safe over-flagged) | 1 |
| True Negatives | 903 |

> Both models miss only **1 hazardous asteroid** across the entire test set — a false negative rate of under **0.2%**.

---

### 🔑 XGBoost Feature Importances (Top 10)

Plotted via `xgboost.plot_importance()` using the default `weight` metric (number of times each feature is used in a split across all trees).

| Rank | Feature | Importance |
|:---:|---|:---:|
| 1 | `Minimum Orbit Intersection` | ████████████ Highest |
| 2 | `Absolute Magnitude` | ██████████ |
| 3 | `Est Dia in KM(min)` | ████████ |
| 4 | `Miss Dist.(Astronomical)` | ██████ |
| 5 | `Relative Velocity km per sec` | █████ |
| 6 | `Orbit Uncertainity` | ████ |
| 7 | `Eccentricity` | ███ |
| 8 | `Semi Major Axis` | ███ |
| 9 | `Inclination` | ██ |
| 10 | `Perihelion Distance` | ██ |

> `Minimum Orbit Intersection` (MOID) is the dominant split feature — it directly measures whether an asteroid's orbital path comes close to Earth's, making it the most physically meaningful hazard proxy in the dataset.

---

### 📌 Why Near-Perfect Scores?

The near-100% accuracy is well-documented for this specific dataset and is physically motivated — not a sign of overfitting:

| Reason | Explanation |
|---|---|
| **MOID is a definitional feature** | NASA uses MOID < 0.05 AU as part of the formal PHA definition — the label is partially derived from this feature |
| **High physical separability** | PHAs are genuinely distinct from safe asteroids in size, brightness, and orbital proximity |
| **Clean, curated labels** | Labels originate from NASA JPL's precise orbital mechanics — minimal label noise |
| **No data leakage** | All redundant columns removed prior to training; split performed after cleaning |

---

## 🏆 Results Summary

| Category | XGBoost | LightGBM |
|---|:---:|:---:|
| **Accuracy** | **99.93%** | 99.86% |
| **PHA Precision** | **1.000** | 0.998 |
| **PHA Recall** | 0.998 | 0.998 |
| **PHA F1-Score** | **0.999** | 0.998 |
| **False Negatives (test set)** | 1 | 1 |
| **False Positives (test set)** | **0** | 1 |
| **Tuning Benefit** | Negligible | Negligible |
| **Training Speed** | Moderate | ⚡ Faster |

> **Winner: XGBoost** — marginally higher precision and zero false positives on the test set. In production at scale, LightGBM's histogram-based algorithm would be preferred for its speed advantage on large streaming datasets (e.g. live Minor Planet Center feeds).

---

## 💡 Key Insights

1. **MOID is the defining predictor.** Minimum Orbit Intersection — the geometric minimum distance between an asteroid's orbit and Earth's — dominates both models' feature importance. This is physically grounded: NASA uses MOID < 0.05 AU as part of the formal PHA criterion, so the label is partially derived from this feature.

2. **Gradient boosting is the right tool here.** Both XGBoost and LightGBM find the near-perfect decision boundary immediately, even on default settings. The non-linear interaction between MOID, magnitude, and diameter is exactly the kind of structure that boosted trees are optimised to exploit.

3. **Tuning had negligible impact.** Setting `n_estimators=100` and `learning_rate=0.1` produced identical test results to default configurations for both models. This is a sign that the dataset is well-structured and the default models are already near-optimal — not that tuning is unimportant in general.

4. **Redundant feature removal was critical for interpretability.** The original 40 columns contained many duplicate-unit representations of the same physical quantity. Removing these doesn't change tree-based model performance, but it produces clean feature importance plots and reduces correlation heatmap noise — both important for communicating findings clearly.

5. **LightGBM vs XGBoost trade-off is context-dependent.** For this dataset, both are equally capable. XGBoost had a slight edge in precision (zero FP vs one FP). LightGBM's leaf-wise growth and histogram binning make it faster at scale — the better default choice for production pipelines processing new asteroid data continuously.

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.10+
- Jupyter Notebook / JupyterLab

### Clone & Install

```bash
git clone https://github.com/Shivang20303/NASA-Asteroid-Classification.git
cd NASA-Asteroid-Classification
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
xgboost>=1.7
lightgbm>=4.0
matplotlib>=3.7
seaborn>=0.13
jupyter>=1.0
```

---

## 🚀 Usage

```bash
jupyter notebook notebook/NASA_Asteroid_Classification.ipynb
```

### Notebook Structure

| Section | Contents |
|---|---|
| **1. Imports & Data Loading** | Library setup, CSV ingestion, `.head()` / `.shape()` / `.info()` |
| **2. Data Cleaning** | Drop IDs, dates, zero-variance columns; encode target with `LabelEncoder` |
| **3. EDA — Correlation Heatmap (Round 1)** | Full 34-feature heatmap to identify redundant columns |
| **4. Feature Reduction** | Drop 11 duplicate-unit columns; describe reduced dataset |
| **5. EDA — Correlation Heatmap (Round 2)** | Cleaner heatmap on 19 final features |
| **6. Train/Test Split** | 70/30 stratified split, `random_state=101` |
| **7. XGBoost — Default** | Train model, plot feature importances |
| **8. LightGBM — Default** | Train model |
| **9. Evaluation (Default params)** | Accuracy, classification report, confusion matrix — both models |
| **10. XGBoost — Tuned** | `n_estimators=100`, `learning_rate=0.1`, `eval_metric='logloss'` |
| **11. LightGBM — Tuned** | `n_estimators=100`, `learning_rate=0.1` |
| **12. Evaluation (Tuned params)** | Accuracy, classification report, confusion matrix — both models |

---

## 🔭 Future Work

| Enhancement | Description |
|---|---|
| **SHAP Explainability** | Per-prediction explanation — critical for building trust in a safety-critical system |
| **Hyperparameter Search** | `Optuna` or `BayesSearchCV` for principled tuning beyond manual `n_estimators` / `lr` |
| **Cross-Validation** | k-fold CV to validate generalisation beyond a single 70/30 split |
| **Threshold Optimisation** | Plot precision-recall curve; tune classification threshold explicitly for recall ≥ 0.999 |
| **Real-Time API Integration** | Wrap the best model in a FastAPI endpoint consuming the [NASA NeoWs live feed](https://api.nasa.gov/) |
| **Imbalance Techniques** | Explore SMOTE / ADASYN as alternatives to relying on natural class separation |
| **Model Drift Monitoring** | Add Evidently AI to detect feature or prediction distribution drift over time |

---

## 📚 References

- NASA Center for Near Earth Object Studies (CNEOS): https://cneos.jpl.nasa.gov/
- NASA NeoWs API: https://api.nasa.gov/
- Kaggle Dataset: https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification
- XGBoost Documentation: https://xgboost.readthedocs.io/
- LightGBM Documentation: https://lightgbm.readthedocs.io/
- Planetary Defense Overview: https://www.nasa.gov/planetarydefense

---

<div align="center">

**Built by [Shivang](https://github.com/Shivang20303)**  
_Data Science & ML Portfolio · NASA Asteroid Hazard Classification_

⭐ If you found this useful, consider giving it a star!

</div>
