"""
Silver Layer ETL Script for Attributes Data

- Validates raw formats (Age, snapshot_date)
- Normalizes Occupation
- Casts Age and snapshot_date
- Checks for nulls, duplicates, temporal alignment
"""

import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lower, regexp_replace, when, trim
from pyspark.sql.types import IntegerType, DateType
import pandas as pd



def create_spark_session():
    return (SparkSession
            .builder
            .appName("Silver Layer - Attributes")
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
        print("No format issues detected.")


def validate_attributes_data(df):
    print("Validating attributes data...")
    invalid_counts = {}

    invalid_age_df = df.filter(~col("Age").rlike("^-?\\d+$"))
    invalid_age_count = invalid_age_df.count()
    if invalid_age_count > 0:
        invalid_counts["Age"] = invalid_age_count
        invalid_age_df.select("Customer_ID", "Age").show(3, truncate=False)

    invalid_date_df = df.filter(~col("snapshot_date").rlike("^\\d{4}-\\d{2}-\\d{2}$"))
    invalid_date_count = invalid_date_df.count()
    if invalid_date_count > 0:
        invalid_counts["snapshot_date"] = invalid_date_count
        invalid_date_df.select("Customer_ID", "snapshot_date").show(3, truncate=False)

    summarize_invalids(invalid_counts)
    return df


def transform_attributes_data(df):
    """
    Transforms attributes data by:
    - Selecting expected columns: Customer_ID, Age, Occupation, snapshot_date
    - Trimming Customer_ID
    - Casting Age to IntegerType if valid
    - Normalizing Occupation:
        1. Convert to lowercase
        2. Replace underscores with spaces
        3. Replace blanks or placeholder '_______' with 'unknown'
    - Casting snapshot_date to DateType
    """
    
    print("Transforming attributes data...")

    # Select expected columns
    df = df.select("Customer_ID", "Age", "Occupation", "snapshot_date")
    df = df.withColumn("Customer_ID", trim(col("Customer_ID")))

    # Age: safe casting
    df = df.withColumn("Age",
                       when(col("Age").rlike("^-?\\d+$"), col("Age").cast(IntegerType()))
                       .otherwise(None))

    # Normalize Occupation
    df = df.withColumn("Occupation", lower(col("Occupation")))
    df = df.withColumn("Occupation", regexp_replace(col("Occupation"), "_", " "))
    df = df.withColumn("Occupation",
                       when(col("Occupation").rlike("^\\s*$"), "unknown")
                       .when(col("Occupation") == "_______", "unknown")
                       .otherwise(col("Occupation")))

    # Date cast
    df = df.withColumn("snapshot_date", to_date(col("snapshot_date")))

    df.printSchema()
    df.show(5, truncate=False)
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
    total_rows = df.count()
    unique_keys = df.select("Customer_ID", "snapshot_date").distinct().count()
    if unique_keys < total_rows:
        print(f"Found {total_rows - unique_keys} duplicate rows.")
    else:
        print("All (Customer_ID, snapshot_date) pairs are unique.")


def check_snapshot_date_continuity(df):
    import pandas as pd
    print("Checking snapshot_date continuity...")
    dates = df.select("snapshot_date").distinct().orderBy("snapshot_date").toPandas()
    all_dates = set(dates["snapshot_date"])
    expected_range = pd.date_range(start=dates["snapshot_date"].min(),
                                   end=dates["snapshot_date"].max(), freq="MS")
    missing = [d for d in expected_range if d not in all_dates]
    if missing:
        print("Missing snapshot dates:")
        print(missing)
    else:
        print("All expected snapshot months are present.")


def summarize_age_stats(df):
    print("Summary stats for Age:")
    stats = df.select("Age").describe().toPandas().set_index("summary")["Age"]
    print(f"Min: {stats['min']}, Max: {stats['max']}, Mean: {stats['mean']}")


def process_attributes_data():
    spark = create_spark_session()
    base_dir = Path(os.getcwd())
    input_dir = base_dir / "datamart" / "bronze" / "attributes"
    output_dir = base_dir / "datamart" / "silver" / "attributes"
    ensure_output_directory(output_dir)

    print(f"Reading data from: {input_dir}")
    bronze_df = spark.read.parquet(str(input_dir))
    print(f"Row count: {bronze_df.count()}")
    bronze_df.show(5)
    bronze_df.printSchema()

    validated_df = validate_attributes_data(bronze_df)
    silver_df = transform_attributes_data(validated_df)

    post_cast_null_check(silver_df)
    check_duplicate_keys(silver_df)
    check_snapshot_date_continuity(silver_df)
    summarize_age_stats(silver_df)

    print(f"Writing cleaned data to: {output_dir}")
    silver_df.write.mode("overwrite").parquet(str(output_dir))
    print("Silver Layer: Attributes processing complete.")
    spark.stop()


if __name__ == "__main__":
    process_attributes_data()
