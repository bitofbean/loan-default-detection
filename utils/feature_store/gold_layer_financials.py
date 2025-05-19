"""
Gold Layer ETL Script for Financials Data

This script transforms silver layer financial data into ML-ready gold features by:
1. Clipping outliers
2. Adding outlier flags
3. Adding missing value flags
4. Creating unknown category flags
5. Creating ratio features
6. Log-transforming skewed features
7. Grouping rare categories

Input: Silver financial data (datamart/silver/financials/)
Output: Gold feature store (datamart/gold/financials/)
"""

import os
import numpy as np
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, log, count, 
    mean, stddev, min, max, 
    expr, udf
)
from pyspark.sql.types import IntegerType, DoubleType


def create_spark_session():
    """Creates and returns a Spark session for gold layer processing."""
    return (SparkSession.builder
            .appName("Gold Layer - Financials")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


def ensure_output_directory(output_path):
    """Creates output directory if it doesn't exist."""
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory: {output_path}")


def clip_outliers(df):
    """
    Clips outlier values for specified numeric columns.
    Creates outlier flags for clipped fields.
    """
    print("Clipping outliers and creating outlier flags...")
    
    # Define outlier thresholds
    outlier_caps = {
        "Annual_Income": 2000000,
        "Num_of_Loan": 20,
        "Num_Bank_Accounts": 20,
        "Num_Credit_Inquiries": 30
    }
    
    # For each field, clip values and create outlier flag
    for field, cap in outlier_caps.items():
        flag_field = f"{field.lower()}_outlier_flag"
        
        # Create outlier flag before clipping
        df = df.withColumn(
            flag_field,
            when(col(field) > cap, 1).otherwise(0)
        )
        
        # Apply clipping
        df = df.withColumn(
            field,
            when(col(field) > cap, cap).otherwise(col(field))
        )
        
        # Summary of clipping effect
        outlier_count = df.filter(col(flag_field) == 1).count()
        total_count = df.count()
        print(f"  {field}: Clipped {outlier_count} values ({outlier_count/total_count:.2%}) at {cap}")
    
    return df


def add_missing_value_flags(df):
    """
    Adds binary flags for columns with non-negligible null values.
    """
    print("Creating missing value flags...")
    
    # Columns to check for nulls
    null_prone_columns = [
        "Amount_invested_monthly",
        "Num_of_Loan",
        "Monthly_Balance"
    ]
    
    # Create missing flags and count nulls
    for field in null_prone_columns:
        flag_field = f"{field.lower()}_missing_flag"
        
        # Add flag
        df = df.withColumn(
            flag_field,
            when(col(field).isNull(), 1).otherwise(0)
        )
        
        # Count and report nulls
        null_count = df.filter(col(field).isNull()).count()
        total_count = df.count()
        print(f"  {field}: {null_count} null values ({null_count/total_count:.2%})")
    
    return df


def add_unknown_category_flags(df):
    """
    Adds binary flags for categorical variables with unknown/not measured values.
    """
    print("Creating unknown category flags...")
    
    # Define mapping of fields and their "unknown" values
    unknown_mappings = {
        "Credit_Mix": ("unknown", "credit_mix_unknown_flag"),
        "Payment_Behaviour": ("unknown", "payment_behaviour_unknown_flag"),
        "Payment_of_Min_Amount": ("Not Measured", "payment_min_not_measured_flag")
    }
    
    # For each field, create appropriate flag
    for field, (unknown_value, flag_field) in unknown_mappings.items():
        df = df.withColumn(
            flag_field,
            when(col(field) == unknown_value, 1).otherwise(0)
        )
        
        # Count and report unknowns
        unknown_count = df.filter(col(field) == unknown_value).count()
        total_count = df.count()
        print(f"  {field}: {unknown_count} '{unknown_value}' values ({unknown_count/total_count:.2%})")
    
    return df


def create_ratio_features(df):
    """
    Creates ratio features to capture relationships between numeric fields.
    """
    print("Creating ratio features...")
    
    # 1. EMI to income ratio (keep existing)
    df = df.withColumn(
        "emi_to_income_ratio",
        when(
            (col("Monthly_Inhand_Salary").isNotNull()) & 
            (col("Monthly_Inhand_Salary") > 0) &
            (col("Total_EMI_per_month").isNotNull()),
            col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")
        ).otherwise(None)
    )
    print("  Created emi_to_income_ratio")
    
    # 2. Debt to income ratio (keep existing)
    df = df.withColumn(
        "debt_to_income_ratio",
        when(
            (col("Annual_Income").isNotNull()) & 
            (col("Annual_Income") > 0) &
            (col("Outstanding_Debt").isNotNull()),
            col("Outstanding_Debt") / col("Annual_Income")
        ).otherwise(None)
    )
    print("  Created debt_to_income_ratio")
    
    # 3. Investment to income ratio (NEW)
    df = df.withColumn(
        "investment_to_income_ratio",
        when(
            (col("Monthly_Inhand_Salary").isNotNull()) & 
            (col("Monthly_Inhand_Salary") > 0) &
            (col("Amount_invested_monthly").isNotNull()),
            col("Amount_invested_monthly") / col("Monthly_Inhand_Salary")
        ).otherwise(None)
    )
    print("  Created investment_to_income_ratio")
    
    # 4. Loan per credit card ratio (NEW)
    df = df.withColumn(
        "loan_per_card_ratio",
        when(
            (col("Num_Credit_Card").isNotNull()) & 
            (col("Num_Credit_Card") > 0) &
            (col("Num_of_Loan").isNotNull()),
            col("Num_of_Loan") / col("Num_Credit_Card")
        ).otherwise(None)
    )
    print("  Created loan_per_card_ratio")
    
    # 5. Credit inquiries per account (NEW)
    df = df.withColumn(
        "credit_inquiries_per_account",
        when(
            (col("Num_Bank_Accounts").isNotNull()) & 
            (col("Num_Bank_Accounts") > 0) &
            (col("Num_Credit_Inquiries").isNotNull()),
            col("Num_Credit_Inquiries") / col("Num_Bank_Accounts")
        ).otherwise(None)
    )
    print("  Created credit_inquiries_per_account")
    
    # 6. Credit limit change ratio (NEW)
    df = df.withColumn(
        "credit_limit_change_ratio",
        when(
            (col("Monthly_Inhand_Salary").isNotNull()) & 
            (col("Monthly_Inhand_Salary") > 0) &
            (col("Changed_Credit_Limit").isNotNull()),
            col("Changed_Credit_Limit") / col("Monthly_Inhand_Salary")
        ).otherwise(None)
    )
    print("  Created credit_limit_change_ratio")
    
    return df


def log_transform_skewed_fields(df):
    """
    Applies log transformation (log(x+1)) to skewed numeric fields.
    """
    print("Log-transforming skewed fields...")
    
    # Fields to log-transform
    skewed_fields = [
        "Annual_Income",
        "Outstanding_Debt"
    ]
    
    # Apply log(x+1) transformation
    for field in skewed_fields:
        log_field = f"log_{field}"
        df = df.withColumn(
            log_field,
            log(col(field) + 1)
        )
    
    return df


def group_rare_categories(df):
    """
    Groups rare categories in Payment_Behaviour to 'other'.
    """
    print("Grouping rare Payment_Behaviour categories...")
    
    # Get frequency distribution of Payment_Behaviour
    value_counts = df.groupBy("Payment_Behaviour").count().collect()
    
    # Identify rare categories (< 100 occurrences)
    rare_categories = [row["Payment_Behaviour"] for row in value_counts if row["count"] < 100]
    
    if rare_categories:
        print(f"  Found {len(rare_categories)} rare categories to group as 'other'")
        
        # Create new column with grouped categories
        df = df.withColumn(
            "Payment_Behaviour_grouped",
            when(col("Payment_Behaviour").isin(rare_categories), "other")
            .otherwise(col("Payment_Behaviour"))
        )
    else:
        print("  No rare categories found to group")
        df = df.withColumn("Payment_Behaviour_grouped", col("Payment_Behaviour"))
    
    return df


def drop_redundant_columns(df):
    """
    Drops columns that are either redundant or not needed for ML.
    """
    print("Dropping redundant columns...")
    
    # Columns to drop
    columns_to_drop = [
        "Credit_History_Age"  # Already parsed into Credit_History_Months
    ]
    
    # Drop columns
    df = df.drop(*columns_to_drop)
    
    return df


def impute_missing_values(df):
    """
    Imputes missing values for selected columns to prevent model errors.
    Uses median for numeric columns.
    """
    print("Imputing missing values...")
    
    # Get aggregate stats for imputation
    stats = df.select([
        mean("Amount_invested_monthly").alias("mean_amount_invested"),
        mean("Monthly_Balance").alias("mean_monthly_balance"),
        mean("Num_of_Loan").alias("mean_num_loans")
    ]).collect()[0]
    
    # Impute numeric columns with means
    df = df.withColumn(
        "Amount_invested_monthly",
        when(col("Amount_invested_monthly").isNull(), stats["mean_amount_invested"])
        .otherwise(col("Amount_invested_monthly"))
    )
    
    df = df.withColumn(
        "Monthly_Balance",
        when(col("Monthly_Balance").isNull(), stats["mean_monthly_balance"])
        .otherwise(col("Monthly_Balance"))
    )
    
    df = df.withColumn(
        "Num_of_Loan",
        when(col("Num_of_Loan").isNull(), stats["mean_num_loans"])
        .otherwise(col("Num_of_Loan"))
    )
    
    return df


def process_financials_gold():
    """Main function to process financials data into gold layer."""
    spark = create_spark_session()
    base_dir = Path(os.getcwd())
    input_path = base_dir / "datamart" / "silver" / "financials"
    output_path = base_dir / "datamart" / "gold" / "financials"
    
    # Ensure output directory exists
    ensure_output_directory(output_path)
    
    # Read silver data
    print(f"Reading silver data from: {input_path}")
    silver_df = spark.read.parquet(str(input_path))
    print(f"Initial row count: {silver_df.count()}")
    silver_df.printSchema()
    
    # Apply gold transformations
    gold_df = silver_df
    gold_df = clip_outliers(gold_df)
    gold_df = add_missing_value_flags(gold_df)
    gold_df = add_unknown_category_flags(gold_df)
    gold_df = create_ratio_features(gold_df)
    gold_df = log_transform_skewed_fields(gold_df)
    gold_df = group_rare_categories(gold_df)
    gold_df = drop_redundant_columns(gold_df)
    
    # Check if we need to impute missing values
    null_counts = {col_name: gold_df.filter(col(col_name).isNull()).count() 
                  for col_name in gold_df.columns if "flag" not in col_name.lower()}
    
    null_fields = [col for col, count in null_counts.items() if count > 0]
    if null_fields:
        print(f"Found nulls in {len(null_fields)} columns: {null_fields}")
        gold_df = impute_missing_values(gold_df)
    
    # Verify row granularity 
    total_rows = gold_df.count()
    unique_keys = gold_df.select("Customer_ID", "snapshot_date").distinct().count()
    
    if total_rows > unique_keys:
        print(f"WARNING: Found {total_rows - unique_keys} duplicate (Customer_ID, snapshot_date) pairs!")
        # For safety, we'll deduplicate by taking the latest row per key
        gold_df = gold_df.orderBy("Customer_ID", "snapshot_date").dropDuplicates(["Customer_ID", "snapshot_date"])
        print(f"After deduplication: {gold_df.count()} rows")
    else:
        print("Row granularity verified: One row per (Customer_ID, snapshot_date)")
    
    # Write gold data
    print(f"Writing gold data to: {output_path}")
    gold_df.write.mode("overwrite").parquet(str(output_path))
    print("Gold Layer: Financials processing complete.")
    
    # Generate summary stats for key features
    print("\nSummary statistics for key features:")
    gold_df.select([
        "Annual_Income", "log_Annual_Income", 
        "Outstanding_Debt", "log_Outstanding_Debt",
        "debt_to_income_ratio", "emi_to_income_ratio", 
        "investment_to_income_ratio", "loan_per_card_ratio",
        "credit_inquiries_per_account", "credit_limit_change_ratio",
        "Credit_Utilization_Ratio", "Credit_History_Months"
    ]).summary().show()
    
    # Show sample of final data
    print("\nSample of gold data:")
    gold_df.select([
        "Customer_ID", "snapshot_date", 
        "Annual_Income", "log_Annual_Income",
        "annual_income_outlier_flag",
        "num_of_loan_missing_flag",
        "emi_to_income_ratio",
        "investment_to_income_ratio",
        "Payment_Behaviour_grouped"
    ]).show(5)
    
    spark.stop()


if __name__ == "__main__":
    process_financials_gold()