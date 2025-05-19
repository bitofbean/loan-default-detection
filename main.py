"""
Main ETL entrypoint — Bronze + Silver + Gold Layer

Executes ETL pipeline using pipeline_config.yaml to dynamically control
which layers and datasets to process.

Supports Medallion Architecture: Bronze → Silver → Gold → Label
"""

import sys
import os
import yaml
from pathlib import Path
from pyspark.sql import SparkSession

# Bronze + Silver for feature store
from utils.feature_store.bronze_layer import ingest_raw_data
from utils.feature_store.silver_layer_clickstream import process_clickstream_data
from utils.feature_store.silver_layer_attributes import process_attributes_data
from utils.feature_store.silver_layer_financials import process_financials_data
from utils.feature_store.silver_layer_loan_daily import process_loan_daily_data

# Gold for feature store
from utils.feature_store.gold_layer_clickstream import process_clickstream_features
from utils.feature_store.gold_layer_attributes import process_attributes_gold
from utils.feature_store.gold_layer_financials import process_financials_gold
from utils.feature_store.gold_layer_loan_daily import process_loan_daily_gold

# Final join
from utils.feature_store.join_feature_label import join_gold_features_and_label

# Label store (external from lab2)
from label_pipeline.label_main import generate_label_store
from label_pipeline.label_main import cleanup_label_store_only


def load_config():
    with open("pipeline_config.yaml", "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    try:
        config = load_config()
        datasets = config.get("datasets", {})

        if config.get("run_bronze", False):
            print("Starting Bronze Layer ETL...")
            os.makedirs("datamart/bronze/lms/", exist_ok=True)
            ingest_raw_data()
            print("Bronze Layer ETL complete.\n")

        if config.get("run_silver", False):
            print("Starting Silver Layer ETL...")
            if datasets.get("clickstream", False):
                print("→ Processing Clickstream data...")
                process_clickstream_data()
            if datasets.get("attributes", False):
                print("→ Processing Attributes data...")
                process_attributes_data()
            if datasets.get("financials", False):
                print("→ Processing Financials data...")
                process_financials_data()
            if datasets.get("loan_daily", False):
                print("→ Processing Loan Daily data...")
                process_loan_daily_data()
            print("Silver Layer ETL complete.\n")

        if config.get("run_gold", False):
            print("Starting Gold Layer ETL...")
            if datasets.get("clickstream", False):
                print("→ Generating Clickstream Features...")
                process_clickstream_features()
            if datasets.get("attributes", False):
                print("→ Generating Attributes Features...")
                process_attributes_gold()
            if datasets.get("financials", False):
                print("→ Generating Financials Features...")
                process_financials_gold()
            if datasets.get("loan_daily", False):
                print("→ Generating Loan Daily Features...")
                process_loan_daily_gold()
            print("Gold Layer ETL complete.\n")

        # Call label store after gold features
        if config.get("run_label", False):
            print("Generating Gold Layer Label Store...")
            generate_label_store()
            print("Label Store Generation Complete.\n")

        if config.get("run_join", False):
            print("Joining Gold Features with Label Store...")
            join_gold_features_and_label()
            print("Final Training Table written to datamart/gold/final_training_table/\n")

    except Exception as e:
        print(f"Error during ETL pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)

    if config.get("run_label_cleanup", False) and not config.get("run_label", False):
        print("Running standalone label cleanup...")
        cleanup_label_store_only()