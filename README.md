# Loan Default Prediction — ETL Pipeline

This repository contains a full end-to-end PySpark pipeline for generating ML-ready training data for loan default prediction. It follows the **Medallion Architecture** (Bronze → Silver → Gold), and supports modular execution of feature and label stores via YAML config toggles.

---

## GitHub Repository

[https://github.com/bitofbean/loan_detection.git](https://github.com/bitofbean/loan_detection.git)

## Project Structure

```bash
loan_default_prediction/
├── main.py                        # Main ETL orchestrator
├── pipeline_config.yaml           # Config to toggle ETL stages
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container build file for Spark + Jupyter
├── docker-compose.yaml            # Docker compose for container orchestration
├── README.md                     # Project documentation
├── README.txt                    # (Local, ignored) School version README
├── .gitignore                    # Files/folders ignored by Git

├── data/                         # Raw input CSV files (ignored by Git)
│   ├── feature_clickstream.csv
│   ├── feature_attributes.csv
│   ├── feature_financials.csv
│   └── lms_loan_daily.csv

├── datamart/                     # Auto-generated ETL output (ignored by Git)
│   ├── bronze/                   # Raw ingestions (Bronze layer)
│   ├── silver/                   # Cleaned and typed data (Silver layer)
│   └── gold/                     # Feature-engineered and labeled data (Gold layer)
│       ├── attributes/
│       ├── clickstream/
│       ├── financials/
│       ├── label_store/          # Partitioned label tables (monthly)
│       ├── loan_daily/
│       └── final_training_table/ # Joined features + labels for ML training

├── label_pipeline/               # Label generation pipeline modules
│   ├── bronze_label_store.py
│   ├── clean_empty_labels_files.py
│   ├── label_bronze_table.py
│   ├── label_silver_table.py
│   ├── label_gold_table.py
│   └── label_main.py

├── utils/                       # Feature store pipeline modules
│   └── feature_store/
│       ├── bronze_layer.py
│       ├── silver_layer_clickstream.py
│       ├── silver_layer_attributes.py
│       ├── silver_layer_financials.py
│       ├── silver_layer_loan_daily.py
│       ├── gold_layer_clickstream.py
│       ├── gold_layer_attributes.py
│       ├── gold_layer_financials.py
│       ├── gold_layer_loan_daily.py
│       └── join_feature_label.py

├── eda/                         # Exploratory notebooks (ignored by Git)
├── docs/                        # Documentation, notes (ignored by Git)
├── presentation/               # Slide decks, images (ignored by Git)
```

---

## Configuration (pipeline_config.yaml)
```bash
run_bronze: true
run_silver: true
run_gold: true
run_label: true
run_label_cleanup: true
run_join: true
```

The label_store: block defines:
- dpd: days past due threshold
- mob: month-on-book to assign label
- snapshot_dates: list of months to generate

## How to Run
```bash
# Build and launch Jupyter/Spark container
docker-compose up --build

# Run full pipeline or partial (based on config)
python main.py
```

---

## Output
After successful run:
- Cleaned and partitioned label store in: `datamart/gold/label_store/`
- Final joined training set in: `datamart/gold/final_training_table/`
- Ready for ML modeling or export to .csv, .parquet, etc.

## Project Status
- Bronze → Silver → Gold ETL validated
- Label store generation and cleanup tested
- Config toggles tested (single-step + full-pipeline mode)
- Final model evaluation (optional)

---

## Git Hygiene
- Ignored via .gitignore:
- data/, datamart/, eda/, docs/, README.txt, presentation/
- Temporary files: .pyc, .log, __pycache__/, .env, .DS_Store