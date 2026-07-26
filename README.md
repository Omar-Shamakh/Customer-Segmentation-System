# Customer Segmentation System

An end-to-end customer segmentation project using K-Means clustering, built on the
[Customer Personality Analysis](https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering)
dataset (2,240 customers, 29 raw features).

## Features
- **Data cleaning & feature engineering** — missing-value imputation, outlier removal,
  and engineered features (`Total_Spending`, `Total_Purchases`, `Family_Size`,
  `Customer_Tenure_Days`, `Total_Campaigns_Accepted`).
- **Exploratory Data Analysis** — see `notebooks/eda.ipynb`.
- **K-Means clustering** with a curated, scaled, one-hot-encoded feature set.
- **Cluster evaluation** via the Elbow Method (WCSS) and Silhouette Score across k=2–10.
- **PCA visualization** of the resulting 5 clusters in 2D.
- **Interactive Streamlit app** for real-time segment prediction on new customer input.
- **Model persistence** — the entire preprocessing + K-Means pipeline is saved as a
  single Joblib artifact (no train/serve skew).

## Segments discovered (k=5)

| Cluster | Profile |
|---|---|
| 0 | Older, budget-conscious households with children — low engagement |
| 1 | Younger, lower-income customers — lowest spenders, low engagement |
| 2 | Mature, comfortable-income customers — strong spenders |
| 3 | High-income, top spenders, highest campaign responsiveness — VIP segment |
| 4 | High-income, high-spending, few/no children — low campaign engagement |

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