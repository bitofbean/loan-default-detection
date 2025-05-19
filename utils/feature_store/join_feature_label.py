from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pathlib import Path

def join_gold_features_and_label():
    # Create Spark session
    spark = SparkSession.builder \
        .appName("Join Gold Features and Label Store") \
        .config("spark.sql.session.timeZone", "UTC") \
        .getOrCreate()

    # Base path
    base_dir = Path.cwd()
    gold_dir = base_dir / "datamart" / "gold"

    # Load all datasets
    def load_df(path_str):
        return spark.read.parquet(str(path_str))

    label_store_path = gold_dir / "label_store"
    label_files = list(label_store_path.glob("*.parquet"))

    if not label_files:
        raise ValueError(f"No label files found in {label_store_path}")

    label_dfs = [spark.read.parquet(str(f)) for f in label_files]
    label_df = label_dfs[0]
    for df in label_dfs[1:]:
        label_df = label_df.unionByName(df)

    print("Loaded label_store with", label_df.count(), "rows")
    label_df.show(5)

    clickstream_df = load_df(gold_dir / "clickstream")
    attributes_df = load_df(gold_dir / "attributes")
    financials_df = load_df(gold_dir / "financials")
    loan_daily_df = load_df(gold_dir / "loan_daily")

    # Extract valid snapshot_dates from label_df
    valid_dates = label_df.select("snapshot_date").distinct()

    # Filter attributes and financials by label snapshot_dates
    attributes_df_filtered = attributes_df.join(valid_dates, on="snapshot_date", how="inner")
    financials_df_filtered = financials_df.join(valid_dates, on="snapshot_date", how="inner")

    # Drop duplicate join keys from right-side table
    loan_daily_clean = loan_daily_df.drop("Customer_ID")

    # Perform joins in order
    final_df = (
        label_df
        .join(clickstream_df, ["Customer_ID", "snapshot_date"], how="left")
        .join(attributes_df_filtered, ["Customer_ID", "snapshot_date"], how="left")
        .join(financials_df_filtered, ["Customer_ID", "snapshot_date"], how="left")
        .join(loan_daily_clean, ["loan_id", "snapshot_date"], how="left")
    )

    # Write final output
    output_path = gold_dir / "final_training_table"
    final_df.write.mode("overwrite").parquet(str(output_path))

    # Optional: summary
    print("Final joined training table written to:", output_path)
    print("Row count:", final_df.count())
    final_df.groupBy("label").count().show()
