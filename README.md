# Loan Default Prediction — ETL Pipeline

This repository contains a full end-to-end PySpark pipeline for generating ML-ready training data for loan default prediction. 
It follows the **Medallion Architecture** (Bronze → Silver → Gold), and supports modular execution of feature and label stores via YAML config toggles.

---

## GitHub Repository

[https://github.com/bitofbean/loan_detection.git](https://github.com/bitofbean/loan_detection.git)

---

## Project Structure

```bash
loan_default_detection/
├── main.py                    # Main ETL orchestrator
├── pipeline_config.yaml       # Config to toggle ETL stages
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container build file for Spark + Jupyter
├── docker-compose.yaml        # Container orchestration setup
├── README.md                  # Project documentation

├── data/                     # Raw input CSVs (ignored by Git)
├── datamart/                 # Auto-generated ETL outputs (ignored by Git)
│   ├── bronze/               # Raw ingestions
│   ├── silver/               # Cleaned and typed data
│   └── gold/                 # Feature-engineered and labeled data
│       ├── attributes/
│       ├── clickstream/
│       ├── financials/
│       ├── label_store/
│       └── loan_daily/

├── label_pipeline/           # Label store pipeline modules
├── utils/                    # Feature store pipeline modules
```

---

## Configuration (pipeline_config.yaml)
```bash
run_bronze: true
run_silver: true
run_gold: true
run_label: true
run_label_cleanup: true
```

The label_store: block defines:
- dpd: days past due threshold
- mob: month-on-book to assign label
- snapshot_dates: list of months to generate

---

## How to Run

Build and launch Jupyter/Spark container
```bash
docker-compose up --build
```

Run full pipeline or partial (based on config)
```bash
python main.py
```

---

## Output
After successful run:
- Cleaned and partitioned label store in: `datamart/gold/label_store/`

---

## Project Status
- Bronze → Silver → Gold ETL validated
- Label store generation and cleanup tested
- Config toggles tested (single-step + full-pipeline mode)

---

## Git Hygiene
- Ignored via .gitignore:
- data/, datamart/, eda/, docs/, README.txt, presentation/
- Temporary files: .pyc, .log, __pycache__/, .env, .DS_Store