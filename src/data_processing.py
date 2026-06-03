import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

def load_data():
    df = pd.read_csv("data/raw/data.csv")
    return df

def create_time_features(df):
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    df["hour"] = df["TransactionStartTime"].dt.hour
    df["day"] = df["TransactionStartTime"].dt.day
    df["month"] = df["TransactionStartTime"].dt.month
    df["year"] = df["TransactionStartTime"].dt.year
    return df

def create_aggregate_features(df):
    agg = (
        df.groupby("CustomerId")
        .agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            std_amount=("Amount", "std"),
            transaction_count=("Amount", "count")
        )
        .reset_index()
    )
    df = df.merge(agg, on="CustomerId", how="left")
    return df

def build_pipeline(df):
    numerical_features = [
        "Amount", "Value", "hour", "day", "month", "year",
        "total_amount", "avg_amount", "std_amount", "transaction_count"
    ]
    categorical_features = [
        "ProductCategory", "ChannelId", "ProviderId", "PricingStrategy"
    ]
    
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    return preprocessor

def create_rfm(df):
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)
    
    recency = (snapshot_date - df.groupby("CustomerId")["TransactionStartTime"].max()).dt.days
    frequency = df.groupby("CustomerId").size()
    monetary = df.groupby("CustomerId")["Amount"].sum()
    
    rfm = pd.concat([recency, frequency, monetary], axis=1)
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    return rfm

def create_proxy_target(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)
    
    print("Cluster Means:\n", rfm.groupby("cluster").mean())
    
    risk_cluster = 2
    rfm["is_high_risk"] = (rfm["cluster"] == risk_cluster).astype(int)
    return rfm

def merge_target(df, rfm):
    df = df.merge(
        rfm[["is_high_risk"]],
        left_on="CustomerId",
        right_index=True
    )
    return df

if __name__ == "__main__":
    # 1. Load and build features
    df = load_data()
    df = create_time_features(df)
    df = create_aggregate_features(df)
    
    # 2. Pipeline setup (Note: you build it here, but you aren't calling fit_transform yet!)
    preprocessor = build_pipeline(df)
    
    # 3. RFM and Clustering Target
    rfm = create_rfm(df)
    rfm = create_proxy_target(rfm)
    
    # 4. Merge target back and save
    df = merge_target(df, rfm)
    df.to_csv("data/processed/processed_data.csv", index=False)
    print("Processed dataset saved.")