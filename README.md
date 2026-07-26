# Customer Segmentation System

An end-to-end customer segmentation project using K-Means clustering, built on the
[Customer Personality Analysis](https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering)
dataset (2,240 customers, 29 raw features).

## Features
- **Data cleaning & feature engineering** — missing-value imputation, outlier removal,
  and engineered features (`Total_Spending`, `Total_Purchases`, `Family_Size`,
  `Customer_Tenure_Days`, `Total_Campaigns_Accepted`).
- **Exploratory Data Analysis** — see `notebooks/eda.ipynb`.
- **K-Means clustering** on a curated, log-transformed, robust-scaled feature set,
  clustered in PCA-reduced space rather than raw feature space.
- **Cluster evaluation** via the Elbow Method (WCSS) and Silhouette Score, compared
  systematically across feature subsets, scalers, and clustering spaces (raw vs. PCA).
- **PCA visualization** of the resulting clusters in 2D — the same space K-Means was
  actually fit on, not a separate approximation.
- **Interactive Streamlit app** for real-time segment prediction on new customer input.
- **Model persistence** — the entire preprocessing + PCA + K-Means pipeline is saved as
  a single Joblib artifact (no train/serve skew).

## Model evaluation

Rather than picking a cluster count by eye, the model was tuned through several rounds
of systematic comparison:

1. **Feature selection** — a lean, curated feature set (dropping low-signal/redundant
   engineered features) outperformed both a wider 10-feature set and the original
   tutorial's feature list.
2. **Clustering space** — clustering directly on **PCA-reduced components**, instead of
   the raw scaled feature space, nearly doubled Silhouette Score by removing correlated
   noise before K-Means sees the data.
3. **Scaler choice** — `RobustScaler` (median/IQR-based) outperformed `StandardScaler`,
   since this dataset has genuine extreme-spender outliers that distort mean/std-based
   scaling.
4. **Final feature set** — re-testing feature inclusion after switching scaler/space
   confirmed `Total_Purchases` adds signal once combined with RobustScaler + PCA.

**Final pipeline:** 9 curated features → log-transform skewed variables → RobustScaler →
PCA (2 components, 61.65% variance explained) → K-Means (k=3).

**Result: Silhouette Score improved from 0.16 → 0.563 across these iterations** (k=2–10
and multiple k=3 alternatives were all compared before finalizing; see
`data/processed/cluster_evaluation.csv` and `data/processed/final_experiment_comparison.csv`
for the full comparisons).

## Segments discovered (k=3)

| Cluster | Profile | Count |
|---|---|---|
| 0 | High income, high spending, but low campaign engagement — established value segment | 1,079 |
| 1 | Highest income, highest spending, highest campaign response — VIP segment | 211 |
| 2 | Lower income, low spending, low engagement — budget segment | 946 |

## Project structure

```
├── data/
│   ├── raw/            # place the raw Kaggle CSV here (not tracked in git)
│   └── processed/      # cleaned data, cluster profiles, evaluation results
├── notebooks/          # exploratory analysis
├── src/
│   ├── data_processing.py   # cleaning + feature engineering
│   ├── clustering.py        # preprocessing, PCA, K-Means, elbow/silhouette evaluation
│   ├── visualization.py     # PCA cluster plots
│   └── train.py             # end-to-end training + model saving
├── app/
│   └── streamlit_app.py     # interactive prediction app
└── models/              # trained pipeline (.joblib), cluster profiles, PCA plot
```

## Setup

```powershell
git clone https://github.com/Omar-Shamakh/Customer-Segmentation-System.git
cd Customer-Segmentation-System
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

To retrain from scratch, download the dataset from Kaggle into
`data/raw/customer_segmentation.csv`, then:

```powershell
python src\train.py
```

## Run the app

```powershell
streamlit run app\streamlit_app.py
```

## Tech stack

Python, pandas, scikit-learn, matplotlib, Streamlit, Joblib