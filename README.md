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

Systematically compared feature subsets and cluster counts using the Elbow Method (WCSS)
and Silhouette Score, rather than picking k by eye. A lean, curated feature set (`Age`,
`Income`, `Total_Spending`, `NumWebPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`,
`Recency`, `Total_Campaigns_Accepted`) with log-transformed skewed variables outperformed
both a wider 10-feature set and the tutorial's original feature list — improving Silhouette
Score from 0.16 to **0.28** at k=3.

## Segments discovered (k=3)

| Cluster | Profile | Count |
|---|---|---|
| 0 | Lower income, low spending, low engagement — budget segment | 973 |
| 1 | Higher income, strong spenders, low campaign response | 1057 |
| 2 | Highest income, highest spenders, highly campaign-responsive — VIP segment | 206 |

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