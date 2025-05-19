"""
Bronze Layer Ingestion Script

This script ingests raw CSV files and saves them as parquet files in the bronze layer of the data lake.
Following the Medallion Architecture pattern, the bronze layer stores raw data with minimal to no transformation.

Input files:
- feature_clickstream.csv
- feature_attributes.csv
- feature_financials.csv
- lms_loan_daily.csv

Output: Parquet files in datamart/bronze/
"""

import os
from pathlib import Path
from pyspark.sql import SparkSession


def create_spark_session():
    """
    Creates and returns a Spark session.
    
    Returns:
        SparkSession: The configured Spark session
    """
    return (SparkSession
            .builder
            .appName("Bronze Layer Ingestion")
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


def ingest_raw_data():
    """
    Reads raw CSV files and writes them as parquet files to the bronze layer.
    
    Steps:
    1. Create Spark session
    2. Define input and output paths
    3. Ensure output directory exists
    4. Read each CSV file and write as parquet
    """
    # Create Spark session
    spark = create_spark_session()
    
    # Define base paths
    base_dir = Path(os.getcwd())
    input_dir = base_dir / "data"
    output_dir = base_dir / "datamart" / "bronze"
    
    # Ensure output directory exists
    ensure_output_directory(output_dir)
    
    # Define mapping of CSV files to bronze table names
    csv_to_table = {
        "feature_clickstream.csv": "clickstream",
        "feature_attributes.csv": "attributes",
        "feature_financials.csv": "financials",
        "lms_loan_daily.csv": "loan_daily"
    }
    
    # Process each CSV file
    for csv_file, table_name in csv_to_table.items():
        input_path = input_dir / csv_file
        output_path = output_dir / table_name
        
        # Read the CSV file
        print(f"Reading {input_path}...")
        df = spark.read.option("header", "true").csv(str(input_path))
        
        # Write the dataframe as parquet
        print(f"Writing to {output_path}...")
        df.write.mode("overwrite").parquet(str(output_path))
        print(f"Successfully wrote {csv_file} to {output_path} as '{table_name}'")
    
    print("Bronze layer ingestion complete.")
    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    ingest_raw_data()