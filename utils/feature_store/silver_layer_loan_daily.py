"""
Silver Layer ETL Script for Loan Daily Data

- Validates date and numeric formats
- Casts to proper data types
- Checks for nulls, duplicate keys, and snapshot gaps
"""

import os
from pathlib import Path
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, trim
from pyspark.sql.types import IntegerType, DoubleType, DateType

def create_spark_session():
    return (SparkSession.builder
            .appName("Silver Layer - Loan Daily")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())

def ensure_output_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory: {path}")

def validate_loan_data(df):
    print("Validating loan daily data...")
    issues = {}
    date_cols = ["loan_start_date", "snapshot_date"]
    int_cols = ["tenure", "installment_num", "loan_amt"]
    float_cols = ["due_amt", "paid_amt", "overdue_amt", "balance"]

    for col_name in date_cols:
        invalid_count = df.filter(~col(col_name).rlike("^\\d{4}-\\d{2}-\\d{2}$")).count()
        if invalid_count > 0:
            issues[col_name] = invalid_count

    for col_name in int_cols:
        invalid_count = df.filter(~col(col_name).rlike("^-?\\d+$")).count()
        if invalid_count > 0:
            issues[col_name] = invalid_count

    for col_name in float_cols:
        invalid_count = df.filter(~col(col_name).rlike("^\\d+(\\.\\d+)?$")).count()
        if invalid_count > 0:
            issues[col_name] = invalid_count

    if issues:
        print("Validation issues found:")
        for k, v in issues.items():
            print(f" - {k}: {v} rows")
    else:
        print("No format issues detected.")
    return df

def transform_loan_data(df):
    print("Transforming data...")
    cols = ["loan_id", "Customer_ID", "loan_start_date", "tenure", "installment_num", "loan_amt",
            "due_amt", "paid_amt", "overdue_amt", "balance", "snapshot_date"]
    df = df.select(cols)
    df = df.withColumn("Customer_ID", trim(col("Customer_ID")))

    for d in ["loan_start_date", "snapshot_date"]:
        df = df.withColumn(d, to_date(col(d)))

    for c in ["tenure", "installment_num", "loan_amt"]:
        df = df.withColumn(c, col(c).cast(IntegerType()))

    for f in ["due_amt", "paid_amt", "overdue_amt", "balance"]:
        df = df.withColumn(f, col(f).cast(DoubleType()))

    df.printSchema()
    df.show(5, truncate=False)
    return df

def post_cast_null_check(df):
    print("Checking for nulls after cast...")
    for c in df.columns:
        null_count = df.filter(col(c).isNull()).count()
        if null_count > 0:
            print(f" - {c}: {null_count} nulls")

def check_duplicate_keys(df):
    print("Checking (Customer_ID, snapshot_date) uniqueness...")
    total = df.count()
    unique_keys = df.select("Customer_ID", "snapshot_date").distinct().count()
    if unique_keys < total:
        print(f" - Duplicates found: {total - unique_keys} rows")
    else:
        print(" - All keys are unique.")

def check_snapshot_continuity(df):
    print("Checking snapshot_date coverage...")
    dates = df.select("snapshot_date").distinct().orderBy("snapshot_date").toPandas()
    if dates.empty:
        print(" - No snapshot dates found.")
        return
    expected = pd.date_range(start=dates["snapshot_date"].min(), 
                             end=dates["snapshot_date"].max(), freq="MS")
    missing = [d for d in expected if d not in set(dates["snapshot_date"])]
    if missing:
        print(" - Missing months:", missing)
    else:
        print(" - No missing months.")

def summarize_numeric_ranges(df):
    print("Summarizing numeric columns:")
    num_cols = ["tenure", "installment_num", "loan_amt", "due_amt", "paid_amt", "overdue_amt", "balance"]
    for col_name in num_cols:
        stats = df.select(col_name).describe().toPandas().set_index("summary")[col_name]
        print(f"{col_name}: min={stats['min']}, max={stats['max']}, mean={stats['mean']}")

def process_loan_daily_data():
    spark = create_spark_session()
    base_dir = Path(os.getcwd())
    input_path = base_dir / "datamart" / "bronze" / "loan_daily"
    output_path = base_dir / "datamart" / "silver" / "loan_daily"
    ensure_output_directory(output_path)

    df = spark.read.parquet(str(input_path))
    print(f"Initial row count: {df.count()}")
    df.show(5)
    df.printSchema()

    df = validate_loan_data(df)
    df = transform_loan_data(df)
    post_cast_null_check(df)
    check_duplicate_keys(df)
    check_snapshot_continuity(df)
    summarize_numeric_ranges(df)

    print(f"Saving cleaned data to {output_path}")
    df.write.mode("overwrite").parquet(str(output_path))
    print("Silver Layer: Loan Daily complete.")
    spark.stop()

if __name__ == "__main__":
    process_loan_daily_data()
