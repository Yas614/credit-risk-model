import pandas as pd

from src.data_processing import (
    create_time_features,
    create_aggregate_features
)
def test_time_features():
    # 1. Create dummy input data with a timestamp string
    mock_df = pd.DataFrame({
        "TransactionStartTime": ["2026-06-03 14:30:00"]
    })
    
    # 2. Run your function
    result_df = create_time_features(mock_df)
    
    # 3. Assert that the expected columns exist and have correct values
    assert "hour" in result_df.columns
    assert "day" in result_df.columns
    assert "month" in result_df.columns
    assert "year" in result_df.columns
    
    # Check if the extracted values are correct
    assert result_df.loc[0, "hour"] == 14
    assert result_df.loc[0, "year"] == 2026


def test_aggregate_features():
    # 1. Create dummy input data with repeating CustomerIds
    mock_df = pd.DataFrame({
        "CustomerId": [101, 101, 102],
        "Amount": [100, 200, 50]
    })
    
    # 2. Run your function
    result_df = create_aggregate_features(mock_df)
    
    # 3. Assert columns exist
    assert "total_amount" in result_df.columns
    assert "transaction_count" in result_df.columns
    
    # Check if aggregation logic is mathematically correct
    customer_101 = result_df[result_df["CustomerId"] == 101]
    assert customer_101["total_amount"].iloc[0] == 300
    assert customer_101["transaction_count"].iloc[0] == 2