"""
Silver Layer ETL Script for Clickstream Data

- Validates format BEFORE casting
- Cleans and casts data
- Checks for nulls and duplicates
- Logs basic summary stats
"""

import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, trim
from pyspark.sql.types import IntegerType, DateType
import pandas as pd

def create_spark_session():
    return (SparkSession
            .builder
            .appName("Silver Layer - Clickstream")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


def ensure_output_directory(output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory: {output_path}")


def summarize_invalids(invalid_dict):
    if invalid_dict:
        print("Invalid value summary:")
        for k, v in invalid_dict.items():
            print(f" - {k}: {v} rows")
    else:
        print("No invalid entries found in features.")


def validate_clickstream_data(df):
    print("Validating raw clickstream data...")
    feature_cols = [f"fe_{i}" for i in range(1, 21)]
    invalid_counts = {}

    # Validate each feature column format
    for col_name in feature_cols:
        invalid_df = df.filter(~col(col_name).rlike("^-?\\d+$"))
        count = invalid_df.count()
        if count > 0:
            invalid_counts[col_name] = count
            invalid_df.select("Customer_ID", col_name).show(3, truncate=False)

    # Validate snapshot_date format
    invalid_dates = df.filter(~col("snapshot_date").rlike("^\\d{4}-\\d{2}-\\d{2}$"))
    invalid_date_count = invalid_dates.count()
    if invalid_date_count > 0:
        print(f"Found {invalid_date_count} invalid snapshot_date entries")
        invalid_dates.select("Customer_ID", "snapshot_date").show(3, truncate=False)

    summarize_invalids(invalid_counts)
    return df


def transform_clickstream_data(df):
    print("Transforming clickstream data...")

    feature_cols = [f"fe_{i}" for i in range(1, 21)]
    cols_to_keep = feature_cols + ["Customer_ID", "snapshot_date"]
    df = df.select(cols_to_keep)

    # Clean Customer_ID
    df = df.withColumn("Customer_ID", trim(col("Customer_ID")))

    # Cast features
    for col_name in feature_cols:
        df = df.withColumn(col_name, col(col_name).cast(IntegerType()))

    # Cast snapshot_date
    df = df.withColumn("snapshot_date", to_date(col("snapshot_date")))

    print("Schema after transformation:")
    df.printSchema()
    return df


def post_cast_null_check(df):
    print("Checking for nulls after casting...")
    nulls = {col_name: df.filter(col(col_name).isNull()).count() for col_name in df.columns}
    for k, v in nulls.items():
        if v > 0:
            print(f"{k}: {v} nulls after cast")
    if all(v == 0 for v in nulls.values()):
        print("No nulls found after casting.")


def check_duplicate_keys(df):
    print("Checking for (Customer_ID, snapshot_date) duplicates...")
    distinct_count = df.select("Customer_ID", "snapshot_date").distinct().count()
    total = df.count()
    if distinct_count < total:
        print(f"Found {total - distinct_count} duplicate rows based on key")
    else:
        print("Keys are unique per customer per snapshot.")


def check_snapshot_date_continuity(df):
    print("Checking snapshot_date coverage...")
    distinct_dates = df.select("snapshot_date").distinct().orderBy("snapshot_date").toPandas()
    all_dates = set(distinct_dates["snapshot_date"])
    expected_range = pd.date_range(start=distinct_dates["snapshot_date"].min(), 
                                   end=distinct_dates["snapshot_date"].max(), freq="MS")
    missing = [d for d in expected_range if d not in all_dates]
    if missing:
        print(f"Missing snapshot dates: {missing}")
    else:
        print("All expected snapshot dates are present.")


def summarize_feature_ranges(df):
    print("Feature range summaries:")
    feature_cols = [f"fe_{i}" for i in range(1, 21)]
    for col_name in feature_cols:
        stats = df.select(col_name).describe().toPandas().set_index("summary")[col_name]
        print(f"{col_name}: min={stats['min']}, max={stats['max']}, mean={stats['mean']}")


def process_clickstream_data():
    import pandas as pd  # Required for snapshot_date continuity check
    spark = create_spark_session()
    base_dir = Path(os.getcwd())
    input_dir = base_dir / "datamart" / "bronze" / "clickstream"
    output_dir = base_dir / "datamart" / "silver" / "clickstream"
    ensure_output_directory(output_dir)

    print(f"Reading bronze data from: {input_dir}")
    bronze_df = spark.read.parquet(str(input_dir))
    print(f"Row count: {bronze_df.count()}")
    bronze_df.show(5)
    bronze_df.printSchema()

    validated_df = validate_clickstream_data(bronze_df)
    silver_df = transform_clickstream_data(validated_df)

    post_cast_null_check(silver_df)
    check_duplicate_keys(silver_df)
    summarize_feature_ranges(silver_df)
    check_snapshot_date_continuity(silver_df)

    print(f"Writing cleaned data to: {output_dir}")
    silver_df.write.mode("overwrite").parquet(str(output_dir))
    print("Silver Layer: Clickstream processing complete.")

    spark.stop()


if __name__ == "__main__":
    process_clickstream_data()
