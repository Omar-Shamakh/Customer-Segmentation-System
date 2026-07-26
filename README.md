# Customer Segmentation System

An end-to-end customer segmentation project using K-Means clustering, built on the
[Customer Personality Analysis](https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering)
dataset (2,240 customers, 29 raw features).

## Features
- **Data cleaning & feature engineering** — missing-value imputation, outlier removal,
  and engineered features (`Total_Spending`, `Total_Purchases`, `Family_Size`,
  `Customer_Tenure_Days`, `Total_Campaigns_Accepted`).
- **Exploratory Data Analysis** — see `notebooks/eda.ipynb`.
- **K-Means clustering** on a curated, log-transformed, scaled feature set (selected via
  systematic comparison, not the full engineered feature list — see Model Evaluation below).
- **Cluster evaluation** via the Elbow Method (WCSS) and Silhouette Score across k=2–10,
  compared across multiple feature-set candidates.
- **PCA visualization** of the resulting clusters in 2D.
- **Interactive Streamlit app** for real-time segment prediction on new customer input.
- **Model persistence** — the entire preprocessing + K-Means pipeline is saved as a
  single Joblib artifact (no train/serve skew).

## Model evaluation

Systematically compared feature subsets, cluster counts, and clustering spaces using the
Elbow Method (WCSS) and Silhouette Score, rather than picking k by eye:

1. A lean, curated 8-feature set (log-transformed skewed variables + scaling) outperformed
   both a wider 10-feature set and the tutorial's original feature list.
2. Clustering directly on **2 PCA-reduced components** (instead of the raw scaled feature
   space) roughly doubled Silhouette Score again — from 0.28 to **0.42** at k=3 — by
   removing correlated noise before K-Means sees the data.

Final approach: 8 curated features → log-transform + scale → PCA (2 components,
55.75% variance explained) → K-Means (k=3). Silhouette Score: **0.42**.

## Segments discovered (k=3)

| Cluster | Profile | Count |
|---|---|---|
| 0 | Low income, low spending, low engagement — budget segment | 976 |
| 1 | High income, high spending, **recent** purchasers, moderate campaign response — active high-value segment | 500 |
| 2 | High income, high spending, but **haven't purchased recently**, low campaign response — at-risk/lapsing high-value segment | 760 |

## Project structure

```
├── data/
│   ├── raw/            # place the raw Kaggle CSV here (not tracked in git)
│   └── processed/      # cleaned data & cluster profiles
├── notebooks/          # exploratory analysis
├── src/
│   ├── data_processing.py   # cleaning + feature engineering
│   ├── clustering.py        # preprocessing, K-Means, elbow/silhouette evaluation
│   ├── visualization.py     # PCA cluster plots
│   └── train.py             # end-to-end training + model saving
├── app/
│   └── streamlit_app.py     # interactive prediction app
└── models/              # trained pipeline (.joblib) + cluster profiles + PCA plot
```

## Setup

```powershell
git clone https://github.com/Omar-Shamakh/Customer-Segmentation-System.git
cd Customer-Segmentation-System
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

To retrain from scratch, download the dataset from Kaggle into `data/raw/customer_segmentation.csv`, then:

```powershell
python src\train.py
```

## Run the app

```powershell
streamlit run app\streamlit_app.py
```

## Tech stack

Python, pandas, scikit-learn, matplotlib, Streamlit, Joblib