"""
Data cleaning and feature engineering for customer segmentation.
"""
import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Load raw customer data from CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, drop irrelevant/constant columns, remove outliers."""
    df = df.copy()

    # Income has missing values in this dataset — impute with median (robust to skew)
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # Z_CostContact / Z_Revenue are constant across all rows — zero information, drop
    constant_cols = [c for c in ["Z_CostContact", "Z_Revenue"] if c in df.columns]
    df = df.drop(columns=constant_cols)

    # Remove implausible outliers (known issues in this dataset: age up to 130+, income up to 600k+)
    current_year = pd.Timestamp.now().year
    df["Age"] = current_year - df["Year_Birth"]
    df = df[df["Age"] <= 90]
    df = df[df["Income"] <= 200000]

    df = df.reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create higher-signal features from raw columns for clustering."""
    df = df.copy()

    # Customer tenure in days since enrollment (recency of relationship, not last purchase)
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    reference_date = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days

    # Household composition
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df["Family_Size"] = df["Total_Children"] + df["Marital_Status"].isin(
        ["Married", "Together"]
    ).astype(int) + 1

    # Total spend across all product categories — the single strongest segmentation signal
    spend_cols = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds",
    ]
    df["Total_Spending"] = df[spend_cols].sum(axis=1)

    # Total purchases across all channels
    purchase_cols = [
        "NumDealsPurchases", "NumWebPurchases",
        "NumCatalogPurchases", "NumStorePurchases",
    ]
    df["Total_Purchases"] = df[purchase_cols].sum(axis=1)

    # Total campaigns accepted — engagement signal
    campaign_cols = [c for c in df.columns if c.startswith("AcceptedCmp")] + ["Response"]
    df["Total_Campaigns_Accepted"] = df[campaign_cols].sum(axis=1)

    # Simplify noisy categorical levels
    df["Education"] = df["Education"].replace(
        {"2n Cycle": "Master", "Basic": "Undergraduate", "Graduation": "Undergraduate"}
    )
    df["Marital_Status"] = df["Marital_Status"].replace(
        {"Married": "Partnered", "Together": "Partnered",
         "Single": "Single", "Divorced": "Single",
         "Widow": "Single", "Alone": "Single",
         "Absurd": "Single", "YOLO": "Single"}
    )

    # Drop raw columns now superseded by engineered features
    df = df.drop(columns=["ID", "Year_Birth", "Dt_Customer", "Kidhome", "Teenhome"])

    return df


def preprocess_pipeline(raw_path: str, processed_path: str) -> pd.DataFrame:
    """Run the full cleaning + feature engineering pipeline and save the result."""
    df = load_data(raw_path)
    df = clean_data(df)
    df = engineer_features(df)
    df.to_csv(processed_path, index=False)
    return df


if __name__ == "__main__":
    result = preprocess_pipeline(
        raw_path="data/raw/customer_segmentation.csv",
        processed_path="data/processed/customer_segmentation_clean.csv",
    )
    print(result.shape)
    print(result.columns.tolist())
    print(result.isnull().sum().sum(), "missing values remain")