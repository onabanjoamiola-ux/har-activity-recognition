# Human Activity Recognition — LightGBM with Windowed Feature Engineering

**3rd place finish on the private leaderboard | Private score: 0.90888 | Public score: 0.91329**

A six-class Human Activity Recognition (HAR) system built on smartphone accelerometer data. This project was part of a Kaggle-style competition run within the MSc Data Science programme at Liverpool John Moores University (Module 7021DATSCI — Machine Learning and Data Mining).

---

## Problem Statement

Classify 10-second accelerometer recordings into one of six activities:

| Activity | Train samples | % of dataset |
|---|---|---|
| Walking | 2,452 | 38.9% |
| Jogging | 1,951 | 30.9% |
| Upstairs | 702 | 11.1% |
| Downstairs | 606 | 9.6% |
| Sitting | 321 | 5.1% |
| Standing | 278 | 4.4% |

The dataset covers 36 participants recorded at 20 Hz. A key challenge was significant class imbalance — walking and jogging dominate while sitting and standing are substantially underrepresented.

**Evaluation metric:**
```
Final Score = Average Accuracy − |Accuracy Split 1 − Accuracy Split 2|
```
This formulation penalises inconsistency between splits as severely as it rewards accuracy — a model that overfits to one half is penalised even if its average is acceptable.

---

## Results

| Model version | CV mean | Test accuracy | Delta | Final score |
|---|---|---|---|---|
| Baseline — Tuned GBM (30 metadata features) | 72.31% | 83.14% | 9.12% | 78.58% |
| LightGBM + engineered features (101 features) | 75.97% | 86.30% | 9.69% | 81.45% |
| LightGBM improved (depth=4) | 81.59% | 86.30% | 12.05% | 84.64% |
| **LightGBM + windowed features (final — 279 features)** | **83.08%** | **90.64%** | **9.36%** | **85.96%** |

**Leaderboard:** Public score 0.91329 → Private score 0.90888 → **3rd place final ranking**

The gap between public and private scores (0.44%) confirms that the model generalised to unseen data rather than overfitting to the public evaluation set.

---

## Per-class recall progression

| Activity | RF Baseline | GBM Metadata | LightGBM + Features | Final Windowed |
|---|---|---|---|---|
| Walking | 90% | 92% | 90% | **97%** |
| Jogging | 97% | 98% | 98% | **98%** |
| Sitting | 100% | 95% | 100% | **100%** |
| Standing | 60% | 86% | 86% | **88%** |
| Upstairs | 46% | 44% | 60% | **65%** |
| Downstairs | 25% | 37% | 65% | **73%** |

The largest gains came from the hardest classes — upstairs and downstairs recall improved dramatically through targeted feature engineering.

---

## Methodology

### 1. Exploratory Data Analysis

PCA applied to the 30 pre-extracted metadata features showed that walking and jogging formed reasonably distinct clusters, while upstairs and downstairs overlapped heavily. This confirmed that pre-extracted statistics alone were insufficient for distinguishing stair activities and motivated feature engineering from raw signals.

### 2. Baseline model comparison

Eight algorithms were evaluated on the 30 metadata features using GroupKFold cross-validation. The key finding was that cross-validation accuracy is a misleading target under this evaluation metric. XGBoost achieved the highest CV score (94.63%) but produced the largest train-test gap (13.40%), resulting in a low final score. This demonstrated that the metric structurally penalises overfitting.

| Model | CV accuracy | Test accuracy | Delta | Final score |
|---|---|---|---|---|
| Logistic Regression | 86.89% | 77.15% | 9.74% | 72.28 |
| KNN (k=5) | 93.50% | 81.13% | 12.37% | 74.94 |
| Decision Tree | 88.73% | 80.92% | 7.81% | 77.02 |
| Random Forest | 93.82% | 82.68% | 11.14% | 77.11 |
| SVM | 90.76% | 81.59% | 9.17% | 77.01 |
| XGBoost | 94.63% | 81.23% | 13.40% | 74.53 |
| Gradient Boosting | 92.38% | 83.09% | 9.29% | 78.45 |
| Tuned GBM | 92.27% | 83.14% | 9.12% | 78.58 |

### 3. Feature engineering from raw signals

Features were extracted from the raw 20 Hz accelerometer time series across four groups:

**Time domain** — energy, RMS, IQR, skewness, kurtosis, zero crossing rate, peak count, mean absolute deviation, and range per axis (x, y, z) and signal magnitude.

**Jerk features** — first derivative of the acceleration signal per axis and magnitude. Stair climbing produces sharp, high-magnitude jerk at each step transition. Level walking produces a smoother, more periodic profile. This distinction is not captured by any of the 30 pre-extracted features.

**Frequency domain via FFT** — dominant frequency and power, spectral energy in low (0-1Hz), mid (1-3Hz), and high (3-10Hz) bands, spectral centroid, bandwidth, and entropy. Walking concentrates energy at 1-2Hz; jogging at 2-3Hz. Upstairs and downstairs occupy similar frequency ranges but distribute spectral energy differently across axes.

**Cross-axis correlations** — Pearson correlations between axis pairs (x-y, x-z, y-z). The body adopts a different tilt angle ascending versus descending stairs, producing distinct co-movement patterns that summary statistics cannot detect.

### 4. Windowed feature extraction

Each 10-second snippet was divided into 3 equal sub-windows (~33 samples, ~1.65 seconds each). The full feature set was extracted per window, then aggregated across windows using mean and standard deviation — producing **279 features per snippet**.

The window standard deviation directly quantifies within-snippet variability, which proved particularly discriminative for stair activities since the acceleration profile at the start of a stair recording differs from the end. Experiments with 5 windows performed worse because shorter windows degraded FFT frequency resolution at low frequencies.

### 5. Model — LightGBM

LightGBM was selected following the baseline comparison. Key hyperparameter decisions were guided by the evaluation metric rather than raw accuracy:

| Hyperparameter | Value | Justification |
|---|---|---|
| n_estimators | 500 | Sufficient for convergence without excess computation |
| max_depth | 4 | Constrains complexity; prevents overfitting to training users |
| num_leaves | 31 | Additional complexity control independent of depth |
| learning_rate | 0.06 | Conservative step size improves generalisation |
| min_child_samples | 40 | Prevents fitting to noise in minority classes |
| reg_lambda | 3.0 | L2 regularisation penalises large leaf weights |
| reg_alpha | 0.1 | L1 regularisation encourages sparse feature usage |
| feature_fraction | 0.8 | 80% of features sampled per tree — reduces tree correlation |
| bagging_fraction | 0.8 | 80% of rows sampled per tree — reduces variance |
| class_weight | balanced | Compensates for sitting/standing underrepresentation |

### 6. Cross-validation strategy

GroupKFold (5 folds) with participant ID as the grouping variable. This ensures all snippets from the same participant remain within a single fold, so validation always tests on unseen participants. A random split would allow the same participant's data to appear in both training and validation folds, inflating performance estimates due to individual movement style consistency.

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python 3 | Primary language |
| LightGBM | Final classifier |
| scikit-learn | Baseline models, GroupKFold, PCA, preprocessing |
| SciPy | FFT, peak detection, statistical features |
| Pandas / NumPy | Data manipulation and feature engineering |
| Matplotlib | Visualisation |

---

## Repository structure

```
har-activity-recognition/
├── har_lightgbm_windowed.ipynb    # Full pipeline — EDA, feature engineering, model, results
├── predictions_sample.csv          # Sample of final Kaggle submission predictions
├── requirements.txt
└── README.md
```

---

## Setup & usage

```bash
pip install lightgbm scikit-learn pandas numpy scipy matplotlib
```

The notebook expects the following data files in the working directory (available on Kaggle under the competition dataset):

- `metadata.csv` / `metadata_test.csv` / `metadata_kaggle.csv`
- `signals.csv` / `signals_test.csv` / `signals_kaggle.csv`

---

## Key findings

Feature engineering contributed more to the final score than any model or hyperparameter change. The jump from 78.58% to 85.96% was driven almost entirely by adding 249 engineered features from raw signal data. This reflects a core principle: a richer feature representation reduces the burden on the model to learn complex decision boundaries, and simpler models generalise better than complex ones applied to impoverished representations.

The jerk and FFT features were specifically effective because they captured the temporal and frequency structure of the accelerometer signal that the pre-extracted statistics discarded. Windowed extraction further improved performance by capturing within-snippet non-stationarity — a 10-second stair recording is not stationary as the participant adjusts pace or posture throughout.

---

## What I would improve

- **User-adaptive calibration** — CV fold accuracy ranged from 0.69 to 0.88 across participants, suggesting user-specific fine-tuning at inference time would reduce inter-participant variability
- **Wavelet decomposition** — may provide superior time-frequency discrimination for stair activities compared to FFT, which assumes stationarity
- **SMOTE** — synthetic oversampling for sitting and standing rather than class weighting alone
- **Deep learning** — LSTM or CNN directly on the raw time series to learn temporal patterns without manual feature engineering

---

## Author

**Amiola Onabanjo**
MSc Data Science — Liverpool John Moores University
[GitHub](https://github.com/onabanjoamiola-ux)
