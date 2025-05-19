"""
Gold Layer ETL Script for Loan Daily Features

This script transforms loan_daily data from silver to gold layer, creating ML-ready features:
- Calculates payment ratios and flags for loan performance
- Adds engineered features while preserving original values
- Ensures no data leakage by only using data available at each snapshot_date
- Maintains row-level granularity (1 row per loan_id, snapshot_date)

Input: Silver loan_daily data from datamart/silver/loan_daily/
Output: Gold feature store to datamart/gold/loan_daily/
"""

import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, log, 
    min as sql_min, round as sql_round
)
from pyspark.sql.types import DoubleType, IntegerType


def create_spark_session():
    """Creates and returns a Spark session for gold layer processing."""
    return (SparkSession
            .builder
            .appName("Gold Layer - Loan Features")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


def ensure_output_directory(output_path):
    """Creates the output directory if it doesn't exist."""
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory: {output_path}")


def engineer_loan_features(df):
    """
    Engineers ML-ready features from loan_daily data:
    - Payment ratio and clipped version
    - Underpayment and overdue flags
    - Loan balance indicators
    - Loan progress metrics
    
    Args:
        df: Silver layer loan_daily DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    print("Engineering loan features...")
    
    # Create payment_ratio = paid_amt / due_amt (carefully handling division by zero)
    df = df.withColumn(
        "payment_ratio",
        when(col("due_amt") > 0, col("paid_amt") / col("due_amt")).otherwise(0)
    )
    
    # Create payment_ratio_clipped = min(payment_ratio, 2.0)
    df = df.withColumn(
        "payment_ratio_clipped",
        when(col("payment_ratio") > 2.0, 2.0).otherwise(col("payment_ratio"))
    )
    
    # Create underpaid_flag = 1 if paid_amt < due_amt else 0
    df = df.withColumn(
        "underpaid_flag",
        when((col("due_amt") > 0) & (col("paid_amt") < col("due_amt")), 1).otherwise(0)
    )
    
    # Create has_overdue_flag = 1 if overdue_amt > 0 else 0
    df = df.withColumn(
        "has_overdue_flag",
        when(col("overdue_amt") > 0, 1).otherwise(0)
    )
    
    # Create large_balance_flag = 1 if balance > 8000
    df = df.withColumn(
        "large_balance_flag",
        when(col("balance") > 8000, 1).otherwise(0)
    )
    
    # Create constant_tenure_flag = 1 if tenure == 10
    df = df.withColumn(
        "constant_tenure_flag",
        when(col("tenure") == 10, 1).otherwise(0)
    )
    
    # Create percent_paid = paid_amt / loan_amt
    df = df.withColumn(
        "percent_paid",
        when(col("loan_amt") > 0, col("paid_amt") / col("loan_amt")).otherwise(0)
    )
    
    # Create log_overdue_amt = log(overdue_amt + 1)
    df = df.withColumn(
        "log_overdue_amt",
        log(col("overdue_amt") + 1)
    )
    
    # Create balance_to_loan_ratio = balance / loan_amt
    df = df.withColumn(
        "balance_to_loan_ratio",
        when(col("loan_amt") > 0, col("balance") / col("loan_amt")).otherwise(0)
    )
    
    # Round float columns to 4 decimal places for consistency
    float_cols = ["payment_ratio", "payment_ratio_clipped", "percent_paid", 
                 "log_overdue_amt", "balance_to_loan_ratio"]
    
    for col_name in float_cols:
        df = df.withColumn(col_name, sql_round(col(col_name), 4))
    
    return df


def check_unique_constraints(df):
    """
    Verifies that the data maintains proper row-level granularity.
    Checks if there's one row per (loan_id, snapshot_date).
    
    Args:
        df: DataFrame to check
        
    Returns:
        True if constraints are met, False otherwise
    """
    total_rows = df.count()
    unique_combinations = df.select("loan_id", "snapshot_date").distinct().count()
    
    if total_rows == unique_combinations:
        print(f"✓ Data has expected granularity: {total_rows} rows, {unique_combinations} unique combinations")
        return True
    else:
        print(f"! Data has unexpected duplicates: {total_rows} rows, {unique_combinations} unique combinations")
        return False


def deduplicate_if_needed(df):
    """
    Deduplicates the DataFrame if necessary to ensure one row per (loan_id, snapshot_date).
    
    Args:
        df: DataFrame to deduplicate
        
    Returns:
        Deduplicated DataFrame
    """
    if check_unique_constraints(df):
        return df
    
    print("Deduplicating data to ensure one row per (loan_id, snapshot_date)...")
    return df.dropDuplicates(["loan_id", "snapshot_date"])


def summarize_feature_stats(df):
    """
    Summarizes the distribution of engineered features.
    """
    print("Feature statistics:")
    
    # Summarize continuous features
    continuous_features = [
        "payment_ratio", "payment_ratio_clipped", "percent_paid", 
        "log_overdue_amt", "balance_to_loan_ratio"
    ]
    
    df.select(continuous_features).summary().show()
    
    # Summarize binary features
    binary_features = [
        "underpaid_flag", "has_overdue_flag", 
        "large_balance_flag", "constant_tenure_flag"
    ]
    
    for feature in binary_features:
        counts = df.groupBy(feature).count().orderBy(feature).collect()
        total = sum(row["count"] for row in counts)
        print(f"{feature} distribution:")
        for row in counts:
            percentage = 100.0 * row["count"] / total
            print(f"  Value {row[feature]}: {row['count']} rows ({percentage:.2f}%)")


def process_loan_daily_gold():
    """
    Main function to process loan_daily data from silver to gold layer.
    """
    # Set up Spark and paths
    spark = create_spark_session()
    base_dir = Path(os.getcwd())
    input_dir = base_dir / "datamart" / "silver" / "loan_daily"
    output_dir = base_dir / "datamart" / "gold" / "loan_daily"
    ensure_output_directory(output_dir)
    
    # Read silver data
    print(f"Reading silver data from: {input_dir}")
    silver_df = spark.read.parquet(str(input_dir))
    print(f"Row count: {silver_df.count()}")
    silver_df.show(5)
    silver_df.printSchema()
    
    # Engineer features
    gold_df = engineer_loan_features(silver_df)
    
    # Ensure data consistency
    gold_df = deduplicate_if_needed(gold_df)
    
    # Display sample and stats
    print("Gold layer schema:")
    gold_df.printSchema()
    print("Sample data:")
    gold_df.show(5)
    
    # Summarize engineered features
    summarize_feature_stats(gold_df)
    
    # Write to gold layer
    print(f"Writing gold data to: {output_dir}")
    gold_df.write.mode("overwrite").parquet(str(output_dir))
    print("Gold Layer: Loan Daily feature store complete.")
    
    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    process_loan_daily_gold()