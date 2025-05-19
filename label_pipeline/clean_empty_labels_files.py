# clean_empty_labels_files.py
from pyspark.sql import SparkSession
from pathlib import Path

def delete_empty_label_partitions(label_dir_path="datamart/gold/label_store"):
    spark = SparkSession.builder.getOrCreate()
    label_dir = Path(label_dir_path)

    if not label_dir.exists() or not label_dir.is_dir():
        print(f"Path does not exist or is not a directory: {label_dir.resolve()}")
        return

    deleted = 0
    inspected = 0

    for folder in label_dir.iterdir():
        if not folder.is_dir():
            continue
        try:
            df = spark.read.parquet(str(folder))
            count = df.count()
            print(f"{folder.name}: {count} rows")
            inspected += 1
            if count == 0:
                for file in folder.glob("*"):
                    file.unlink()
                folder.rmdir()
                print(f"Deleted empty label partition: {folder.name}")
                deleted += 1
        except Exception as e:
            print(f"Skipped {folder.name}: {e}")

    print(f"\nInspected: {inspected} label partitions")
    print(f"Cleanup complete. {deleted} empty label folders removed.")

# Allow script to be run standalone
if __name__ == "__main__":
    delete_empty_label_partitions()
