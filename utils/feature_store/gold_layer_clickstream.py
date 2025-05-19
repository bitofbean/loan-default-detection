"""
Gold Layer ETL Script for Clickstream Features

This script reads cleaned clickstream data from the silver layer and performs feature engineering:
- Efficiently computes mean and standard deviation for all features in a single operation
- Preserves original values in *_raw columns
- Identifies and clips outliers (outside 3-sigma range)
- Adds binary flags for outlier detection
- Writes ML-ready feature store to the gold layer
"""

import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, when, lit
from functools import reduce
import pyspark.sql.functions as F


def create_spark_session():
    """
    Creates and returns a Spark session.
    
    Returns:
        SparkSession: Configured Spark session
    """
    return (SparkSession
            .builder
            .appName("Gold Layer - Clickstream Features")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


def ensure_output_directory(output_path):
    """
    Creates the output directory if it doesn't exist.
    
    Args:
        output_path (Path): The output directory path
    """
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory: {output_path}")


def transform_features(df):
    """
    Transforms clickstream features by:
    1. Efficiently computing mean and standard deviation for all features
    2. Preserving original values before clipping
    3. Clipping outliers beyond 3 standard deviations
    4. Adding binary flags for clipped outliers
    
    Args:
        df (DataFrame): Input DataFrame with cleaned clickstream data
        
    Returns:
        DataFrame: Transformed DataFrame with engineered features
    """
    print("Transforming clickstream features...")
    feature_cols = [f"fe_{i}" for i in range(1, 21)]
    
    # 1. Compute all statistics in a single operation
    # Create a list of aggregation expressions for all features
    agg_expressions = []
    for col_name in feature_cols:
        agg_expressions.append(mean(col(col_name)).alias(f"{col_name}_mean"))
        agg_expressions.append(stddev(col(col_name)).alias(f"{col_name}_stddev"))
    
    # Execute single aggregation operation
    stats_df = df.agg(*agg_expressions).collect()[0]
    
    # Build statistics dictionary from results
    stats = {}
    for col_name in feature_cols:
        stats[col_name] = {
            "mean": stats_df[f"{col_name}_mean"],
            "stddev": stats_df[f"{col_name}_stddev"]
        }
        print(f"{col_name}: mean={stats[col_name]['mean']:.2f}, stddev={stats[col_name]['stddev']:.2f}")
    
    # 2. Store original values and add outlier detection
    for col_name in feature_cols:
        # First, preserve original values
        df = df.withColumn(f"{col_name}_raw", col(col_name))
        
        mean_val = stats[col_name]["mean"]
        std_val = stats[col_name]["stddev"]
        
        # Skip outlier detection if stddev is null or zero
        if std_val is None or std_val == 0:
            print(f"Warning: Skipping outlier detection for {col_name} (stddev is {std_val})")
            df = df.withColumn(f"{col_name}_is_outlier", lit(0))
            continue
            
        # Calculate lower and upper bounds (3-sigma)
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
        
        # Create outlier indicator (1 if outlier, 0 if not)
        df = df.withColumn(
            f"{col_name}_is_outlier",
            when(
                (col(col_name) < lower_bound) | (col(col_name) > upper_bound),
                lit(1)
            ).otherwise(lit(0))
        )
        
        # Clip values outside the range [mean - 3*std, mean + 3*std]
        df = df.withColumn(
            col_name,
            when(col(col_name) < lower_bound, lower_bound)
            .when(col(col_name) > upper_bound, upper_bound)
            .otherwise(col(col_name))
        )
    
    # Display schema and sample of transformed data
    print("Schema after transformation:")
    df.printSchema()
    df.show(5, truncate=False)
    
    return df


def process_clickstream_features():
    """
    Main function to process clickstream data:
    1. Read from silver layer
    2. Transform features
    3. Write to gold layer
    """
    spark = create_spark_session()
    base_dir = Path(os.getcwd())
    input_dir = base_dir / "datamart" / "silver" / "clickstream"
    output_dir = base_dir / "datamart" / "gold" / "clickstream"
    
    # Ensure output directory exists
    ensure_output_directory(output_dir)
    
    print(f"Reading silver data from: {input_dir}")
    silver_df = spark.read.parquet(str(input_dir))
    print(f"Input row count: {silver_df.count()}")
    silver_df.show(5)
    silver_df.printSchema()
    
    # Transform features and add outlier detection
    gold_df = transform_features(silver_df)
    
    # Calculate final column counts by type
    feature_count = len([col for col in gold_df.columns if col.startswith("fe_") and not col.endswith("_raw") and not col.endswith("_is_outlier")])
    raw_count = len([col for col in gold_df.columns if col.endswith("_raw")])
    outlier_count = len([col for col in gold_df.columns if col.endswith("_is_outlier")])
    
    print(f"Output row count: {gold_df.count()}")
    print(f"Features: {feature_count}, Raw values: {raw_count}, Outlier flags: {outlier_count}")
    
    # Write to gold layer
    print(f"Writing to gold layer: {output_dir}")
    gold_df.write.mode("overwrite").parquet(str(output_dir))
    print("Gold Layer: Clickstream Features complete.")
    
    spark.stop()


if __name__ == "__main__":
    process_clickstream_features()