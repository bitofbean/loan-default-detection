"""
Gold Layer ETL Script for Attributes Data

This script:
1. Reads the silver layer attributes data
2. Applies feature engineering:
   - Clips Age values to valid range [18, 90]
   - Creates flags for age outliers, missing age, and unknown occupation
3. Writes ML-ready gold layer data

Input: Silver attributes data from datamart/silver/attributes/
Output: Gold attributes data to datamart/gold/attributes/
"""

import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnull, lit
from pyspark.sql.types import IntegerType


def create_spark_session():
    """Creates and returns a Spark session for Gold layer processing."""
    return (SparkSession
            .builder
            .appName("Gold Layer - Attributes")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


def ensure_output_directory(output_path):
    """Creates the output directory if it doesn't exist."""
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory: {output_path}")


def engineer_attributes_features(df):
    """
    Applies feature engineering to attributes data:
    1. Clips Age to valid range [18, 90]
    2. Creates flags for outliers, missing values, and unknown occupation
    
    Args:
        df: Silver layer attributes DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    print("Applying feature engineering to attributes data...")
    
    # Create age_outlier_flag before clipping
    df = df.withColumn(
        "age_outlier_flag",
        when((col("Age").isNotNull()) & 
             ((col("Age") < 18) | (col("Age") > 90)), 1).otherwise(0)
    )
    
    # Clip Age to valid range [18, 90]
    df = df.withColumn(
        "Age",
        when((col("Age").isNotNull()) & 
             ((col("Age") < 18) | (col("Age") > 90)), None)
        .otherwise(col("Age"))
    )
    
    # Create age_missing_flag
    df = df.withColumn(
        "age_missing_flag",
        when(col("Age").isNull(), 1).otherwise(0)
    )
    
    # Create unknown_occupation_flag
    df = df.withColumn(
        "unknown_occupation_flag",
        when(col("Occupation") == "unknown", 1).otherwise(0)
    )
    
    return df


def process_attributes_gold():
    """
    Main function to process attributes data from silver to gold layer.
    """
    # Set up Spark and paths
    spark = create_spark_session()
    base_dir = Path(os.getcwd())
    input_dir = base_dir / "datamart" / "silver" / "attributes"
    output_dir = base_dir / "datamart" / "gold" / "attributes"
    ensure_output_directory(output_dir)
    
    # Read silver data
    print(f"Reading silver data from: {input_dir}")
    silver_df = spark.read.parquet(str(input_dir))
    print(f"Row count: {silver_df.count()}")
    silver_df.show(5)
    silver_df.printSchema()
    
    # Apply feature engineering
    gold_df = engineer_attributes_features(silver_df)
    
    # Display results
    print("Gold layer schema:")
    gold_df.printSchema()
    print("Sample data:")
    gold_df.show(5)
    
    # Count nulls and flags
    print("Flag statistics:")
    gold_df.groupBy().sum("age_outlier_flag", "age_missing_flag", "unknown_occupation_flag").show()
    
    # Write to gold layer
    print(f"Writing gold data to: {output_dir}")
    gold_df.write.mode("overwrite").parquet(str(output_dir))
    print("Gold Layer: Attributes processing complete.")
    
    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    process_attributes_gold()