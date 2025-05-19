# label_main.py
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType

import label_pipeline.label_bronze_table
import label_pipeline.label_silver_table
import label_pipeline.label_gold_table

import yaml
from label_pipeline.clean_empty_labels_files import delete_empty_label_partitions


def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)
    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        current_date = datetime(current_date.year + 1, 1, 1) if current_date.month == 12 else datetime(current_date.year, current_date.month + 1, 1)
    return first_of_month_dates


def generate_label_store():
    spark = SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Config
    start_date_str = "2023-01-01"
    end_date_str = "2024-12-01"
    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

    bronze_dir = "datamart/bronze/lms/"
    silver_dir = "datamart/silver/lms/"
    label_dir = "datamart/gold/label_store/"

    os.makedirs(bronze_dir, exist_ok=True)
    os.makedirs(silver_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for date_str in dates_str_lst:
        label_pipeline.label_bronze_table.process_bronze_table(date_str, bronze_dir, spark)
        label_pipeline.label_silver_table.process_silver_table(date_str, bronze_dir, silver_dir, spark)
        label_pipeline.label_gold_table.process_labels_gold_table(date_str, silver_dir, label_dir, spark, dpd=30, mob=6)

    files_list = [os.path.join(label_dir, os.path.basename(f)) for f in glob.glob(os.path.join(label_dir, '*.parquet'))]
    if files_list:
        df = spark.read.option("header", "true").parquet(*files_list)
        print("Label row_count:", df.count())
        df.show()
    else:
        print("No label files found in label_store/")

    # Load config to check if cleanup is enabled
    with open("pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if config.get("run_label_cleanup", False):
        delete_empty_label_partitions()

def cleanup_label_store_only():
    with open("pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if config.get("run_label_cleanup", False):
        delete_empty_label_partitions()
