"""
Silver Layer ETL Script for Financials Data

- Validates formats for float, int, and date fields
- Parses complex strings (Credit_History_Age)
- Cleans and normalizes categorical variables
- Checks for nulls, duplicates, and date gaps
"""

import os
import re
from pathlib import Path
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, lower, regexp_replace,
    when, trim, length, udf
)
from pyspark.sql.types import IntegerType, DoubleType, DateType

def create_spark_session():
    return (SparkSession.builder
            .appName("Silver Layer - Financials")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())

def ensure_output_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory: {path}")

def parse_credit_history_age(credit_history_age):
    if not credit_history_age or credit_history_age == "_":
        return None
    try:
        years = re.search(r"(\d+)\s*Years?", credit_history_age)
        months = re.search(r"(\d+)\s*Months?", credit_history_age)
        return (int(years.group(1)) if years else 0) * 12 + (int(months.group(1)) if months else 0)
    except:
        return None

def validate_financials_data(df):
    print("Validating financials data...")
    issues = {}
    float_cols = ["Annual_Income", "Monthly_Inhand_Salary", "Changed_Credit_Limit",
                  "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
                  "Amount_invested_monthly", "Monthly_Balance"]
    int_cols = ["Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
                "Delay_from_due_date", "Num_of_Delayed_Payment", "Num_Credit_Inquiries"]

    for col_name in float_cols:
        invalid_count = df.filter(~regexp_replace(col(col_name), "_$", "").rlike("^[0-9]+(\\.[0-9]+)?$"))\
                         .count()
        if invalid_count > 0:
            issues[col_name] = invalid_count

    for col_name in int_cols:
        invalid_count = df.filter(~col(col_name).rlike("^-?[0-9]+$"))\
                         .count()
        if invalid_count > 0:
            issues[col_name] = invalid_count

    date_issues = df.filter(~col("snapshot_date").rlike("^\\d{4}-\\d{2}-\\d{2}$")).count()
    if date_issues > 0:
        issues["snapshot_date"] = date_issues

    if issues:
        print("Invalid field counts:")
        for k, v in issues.items():
            print(f" - {k}: {v}")
    else:
        print("No format issues detected.")
    return df

def transform_financials_data(df):
    print("Transforming financials data...")
    parse_udf = udf(parse_credit_history_age, IntegerType())
    df = df.withColumn("Customer_ID", trim(col("Customer_ID")))

    float_cols = ["Annual_Income", "Monthly_Inhand_Salary", "Changed_Credit_Limit",
                  "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
                  "Amount_invested_monthly", "Monthly_Balance"]
    for col_name in float_cols:
        df = df.withColumn(col_name, regexp_replace(col(col_name), "_$", ""))
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))

    int_cols = ["Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
                "Delay_from_due_date", "Num_of_Delayed_Payment", "Num_Credit_Inquiries"]
    for col_name in int_cols:
        df = df.withColumn(col_name, col(col_name).cast(IntegerType()))

    df = df.withColumn("Credit_Mix", when((col("Credit_Mix") == "_") | (length(trim(col("Credit_Mix"))) == 0), "unknown")
                       .otherwise(lower(col("Credit_Mix"))))

    df = df.withColumn("Payment_Behaviour", 
        when(col("Payment_Behaviour").rlike("^[a-zA-Z0-9_\\s-]+$"), 
             lower(regexp_replace(col("Payment_Behaviour"), "_", " ")))
        .otherwise("unknown"))

    df = df.withColumn("Type_of_Loan", 
        when(col("Type_of_Loan").isNull() | (col("Type_of_Loan") == "") | (col("Type_of_Loan") == "_"), 
             "Not Specified")
        .otherwise(col("Type_of_Loan")))

    df = df.withColumn("Credit_History_Months", parse_udf(col("Credit_History_Age")))

    df = df.withColumn("Payment_of_Min_Amount", 
        when(col("Payment_of_Min_Amount") == "NM", "Not Measured")
        .when(col("Payment_of_Min_Amount").isNull() | (col("Payment_of_Min_Amount") == "") | (col("Payment_of_Min_Amount") == "_"), "Unknown")
        .otherwise(col("Payment_of_Min_Amount")))

    df = df.withColumn("snapshot_date", to_date(col("snapshot_date")))
    df.printSchema()
    return df

def post_cast_null_check(df):
    print("Null values after transformation:")
    for col_name in df.columns:
        null_count = df.filter(col(col_name).isNull()).count()
        if null_count > 0:
            print(f" - {col_name}: {null_count}")

def check_duplicate_keys(df):
    print("Checking (Customer_ID, snapshot_date) duplicates...")
    total = df.count()
    distinct_keys = df.select("Customer_ID", "snapshot_date").distinct().count()
    if total > distinct_keys:
        print(f" - Duplicate rows found: {total - distinct_keys}")
    else:
        print(" - Keys are unique.")

def check_snapshot_date_continuity(df):
    print("Checking snapshot date continuity...")
    dates = df.select("snapshot_date").distinct().orderBy("snapshot_date").toPandas()
    if dates.empty:
        print(" - No snapshot dates found.")
        return
    expected = pd.date_range(start=dates["snapshot_date"].min(), 
                             end=dates["snapshot_date"].max(), freq="MS")
    missing = [d for d in expected if d not in set(dates["snapshot_date"])]
    if missing:
        print(" - Missing snapshot months:", missing)
    else:
        print(" - All snapshot months present.")

def summarize_feature_ranges(df):
    print("Summarizing numeric ranges...")
    num_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, (DoubleType, IntegerType)) and f.name != "Credit_History_Months"]
    for col_name in num_cols:
        stats = df.select(col_name).describe().toPandas().set_index("summary")[col_name]
        print(f"{col_name}: min={stats['min']}, max={stats['max']}, mean={stats['mean']}")

def process_financials_data():
    spark = create_spark_session()
    base_dir = Path(os.getcwd())
    input_path = base_dir / "datamart" / "bronze" / "financials"
    output_path = base_dir / "datamart" / "silver" / "financials"
    ensure_output_directory(output_path)

    df = spark.read.parquet(str(input_path))
    print(f"Initial row count: {df.count()}")
    df.show(5)
    df.printSchema()

    df = validate_financials_data(df)
    df = transform_financials_data(df)
    post_cast_null_check(df)
    check_duplicate_keys(df)
    check_snapshot_date_continuity(df)
    summarize_feature_ranges(df)

    print(f"Saving to: {output_path}")
    df.write.mode("overwrite").parquet(str(output_path))
    print("Silver Layer: Financials complete.")
    spark.stop()

if __name__ == "__main__":
    process_financials_data()