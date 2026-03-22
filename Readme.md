<div align="center">

<img src="https://img.shields.io/badge/✈️-Flight%20Advisor-1A3557?style=for-the-badge&labelColor=1A3557&color=2563A8" alt="Flight Advisor"/>

# Flight Advisor
### Intelligent Platform for Analysis and Prediction of Air Delays

*Tech Challenge — Phase 03 | FIAP Machine Learning Engineering*

---

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Qwen%203-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Dash](https://img.shields.io/badge/Dash-FF4B4B?style=flat-square&logo=Dash&logoColor=white)](https://dash.ploty.com)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-189ABB?style=flat-square)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

<br/>

> **"This flight has a 71% chance of being delayed. Peak hours at JFK (+24%) and the airline's Friday history (+18%) are the main factors. I recommend the 6:00 AM flight — only a 23% risk."**
>
> *— Flight Advisor, via RAG + Qwen 3*

</div>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Installation and Usage](#-installation-and-usage)
- [Notebooks](#-notebooks)
- [Results](#-results)
- [Market Vision](#-market-vision)
- [Limitations and Next Steps](#-limitations-and-next-steps)
- [Author](#-author)

---

## 🎯 About the Project

**Flight Advisor** goes beyond an academic Machine Learning project — it is a functional product with a production-level architecture that solves a real problem: **passengers do not have access to explainable delay predictions before choosing a flight**.

The project combines:

- **Supervised Machine Learning** to predict the probability of delay for each flight
- **Explainability with SHAP** to understand *why* a flight has a high risk
- **Clustering and PCA** to discover hidden patterns in airports and routes
- **RAG with Qwen 3** (Hugging Face Spaces) to answer questions in natural language
- **REST API + Dashboard** to deliver value to the end-user

### The Problem

| | Current Market | Flight Advisor |
|---|---|---|
| **Available Information** | "Delta has 78% on-time performance on this route" | "This specific flight has a 71% chance of being delayed" |
| **Explainability** | None | Top factors with % contribution (SHAP) |
| **Recommendation** | None | "Fly at 6:00 AM — 23% risk" |
| **Interface** | Static tables | Natural language via RAG + LLM |

---

## 🚀 Demo

```bash
# Start the API
uvicorn src.api.main:app --reload

# Start the dashboard
python dashboard/app.py
```

**Example API query:**

```bash
curl -X POST "http://localhost:8000/advise" \
  -H "Content-Type: application/json" \
  -d '{
    "origin_airport": "JFK",
    "destination_airport": "LAX",
    "airline": "DL",
    "month": 11,
    "day_of_week": 5,
    "scheduled_departure": 1800,
    "question": "Is this flight worth it, or is there a better time?"
  }'
```

**Response:**

```json
{
  "delay_probability": 0.71,
  "risk_level": "HIGH",
  "top_factors": [
    {"feature": "peak_hour_JFK", "impact": "+24%"},
    {"feature": "carrier_history_friday", "impact": "+18%"},
    {"feature": "route_congestion", "impact": "+15%"}
  ],
  "advice": "This flight has a HIGH risk of delay (71%). Fridays at 6:00 PM at JFK are historically problematic on the route to LAX. I recommend the 6:00 AM flight — same route and airline, with only a 23% risk. My recommendation: opt for the morning flight if your schedule allows."
}
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FLIGHT ADVISOR                          │
├──────────────┬──────────────────────┬───────────────────────┤
│  ML Pipeline │    RAG Engine        │    Delivery           │
│              │                      │                       │
│ ┌──────────┐ │  ┌────────────────┐  │  ┌──────────────────┐ │
│ │ Random   │ │  │  Vector Store  │  │  │      Dash        │ │
│ │ Forest   │ │  │  FAISS + HF    │  │  │   Dashboard      │ │
│ │ XGBoost  │ │  │  Embeddings    │  │  └────────┬─────────┘ │
│ │ Logistic │ │  └───────┬────────┘  │           │           │
│ └────┬─────┘ │          │           │  ┌──────────────────┐ │
│      │       │  ┌───────▼────────┐  │  │   FastAPI        │ │
│ ┌──────────┐ │  │   Qwen 3       │  │  │   REST API       │ │
│ │  SHAP    │→│  │ HF Spaces      │→ │  └──────────────────┘ │
│ │Explainer │ │  │ (open-source)  │  │                       │
│ └──────────┘ │  └────────────────┘  │                       │
│              │                      │                       │
│ ┌──────────┐ │  ┌────────────────┐  │                       │
│ │ K-Means  │ │  │ Docs: routes,  │  │                       │
│ │   PCA    │ │  │ airports,      │  │                       │
│ │ IsoForest│ │  │ patterns, EDA  │  │                       │
│ └──────────┘ │  └────────────────┘  │                       │
└──────────────┴──────────────────────┴───────────────────────┘
```

### Query Flow

```
User ──► FastAPI ──► ML Model (prob: 71%, risk: HIGH)
                │
                ├──► SHAP Explainer (top factors)
                │
                ├──► RAG Retriever (historical route context)
                │
                └──► Qwen 3 / HF Spaces (natural language response)
                            │
                            └──► User receives an explained recommendation
```

---

## ✨ Features

### 📊 Data Exploration (EDA)
- Descriptive statistics and delay distribution
- Analysis by airline, airport, route, and state
- Interactive geographical maps of US delays (Plotly)
- Seasonal patterns: time of day, day of the week, month, season

### 🤖 Supervised Machine Learning
- **Classification**: predict if a flight will be delayed (>15 min)
- Three models compared: Random Forest, XGBoost, Logistic Regression
- **Feature engineering**: time of day, seasons, holidays, historical delay rate per route
- Metrics: F1-Score, ROC-AUC, Confusion Matrix, Classification Report

### 🔵 Unsupervised Machine Learning
- **K-Means**: clustering of airports by delay profile
- **PCA**: dimensionality reduction for 2D visualization of clusters
- Interpretation of each cluster with a detailed profile

### 🔍 Explainability
- **SHAP values** per individual prediction
- Global feature importance plots
- Answers the question: *"Why does this flight have a high risk?"*

### 🚨 Anomaly Detection
- **Isolation Forest** to identify flights outside the historical pattern
- Visualization of outliers vs. normal behavior

### 🧠 RAG with Qwen 3 (Hugging Face Spaces)
- FAISS vector store indexing historical data of routes and airports
- Embeddings via `sentence-transformers` (local, no API cost)
- Qwen 3 LLM hosted on Hugging Face Spaces (open-source, free)
- Natural language responses with actionable recommendations

### 📡 REST API (FastAPI)
- `/advise` endpoint for a complete query (ML + RAG + LLM)
- `/predict` endpoint for isolated prediction
- Automatic documentation via Swagger UI (`/docs`)

📈 Interactive Dashboard
- Filters by airline, route, period
- Real-time KPIs
- Interactive maps and charts
- Interface to query the Flight Advisor

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.11 | Core of the project |
| **ML** | Scikit-learn, XGBoost | Supervised and unsupervised models |
| **Explainability** | SHAP | Interpretability per prediction |
| **Anomaly** | Isolation Forest | Detection of anomalous flights |
| **RAG** | LangChain + FAISS | Indexing and semantic retrieval |
| **LLM** | Qwen 3 (Hugging Face Spaces) | Generation of natural language responses |
| **Embeddings** | sentence-transformers | Local embeddings with no API cost |
| **API** | FastAPI | REST endpoints for integration |
| **Dashboard** | Dash | Interactive interface |
| **Visualization** | Plotly, Seaborn, Matplotlib | Charts and geographical maps |
| **Data** | Pandas, NumPy | Manipulation and analysis |
| **Versioning** | Git + DVC | Code and models |

---
```
flight-advisor/
│
├── data/
│   ├── raw/                    # Raw data (not versioned)
│   ├── processed/              # Processed data
│   └── vector_store/           # FAISS index for RAG
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploration and visualizations
│   ├── 02_feature_engineering.ipynb    # Feature creation
│   ├── 03_supervised_models.ipynb      # Supervised ML + SHAP
│   ├── 04_unsupervised_models.ipynb    # Clustering + PCA
│   └── 05_anomaly_detection.ipynb      # Anomaly Detection
│
├── src/
│   ├── preprocessing.py        # Cleaning and transformation pipeline
│   ├── model.py                # Model inference and serialization
│   ├── trainer.py              # Training loop and model export
│   ├── explainer.py            # SHAP values and visualizations
│   ├── rag/
│   │   ├── indexer.py          # Document and vector store generation
│   │   ├── retriever.py        # Semantic search in FAISS
│   │   └── advisor.py          # ML + RAG + Qwen 3 orchestrator
│   ├── aws/
│   │   ├── uploader.py         # Upload models and data to S3
│   │   ├── refiner.py          # Cloud refinement orchestration
│   │   └── athena_client.py    # Athena query abstraction
│   ├── jobs/
│   │   ├── weekly_predict.py   # Weekly prediction job
│   │   └── scheduler.py        # Job scheduling (cron / trigger)
│   └── api/
│       └── main.py             # FastAPI endpoints
│
├── dashboard/
│   ├── app.py                  # Dash dashboard
│   ├── build_dashboard_dataset.py # Build slim S3 dataset for dashboard
│   └── parquet_writer.py       # Legacy partition writer (optional)
│
├── models/                     # Serialized models (.pkl)
│
├── infra/
│   ├── s3_config.py            # Bucket names, prefixes and credentials
│   └── athena_schema.sql       # Athena table schema documentation
│
├── requirements.txt
├── .env.example
├── .gitignore
├── PROTOTYPE.md
└── README.md
```
---

## ⚙️ Installation and Usage

### Prerequisites

- Python 3.11+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/GuilhermeLossio/flight-advisor.git
cd flight-advisor
```

### 2. Create the virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit the .env file with your configurations (HuggingFace token, etc.)
```

Optional but recommended for AWS/S3:

- `S3_BUCKET` — default bucket for dataset reads/writes and artifact uploads
- `S3_RAW_PREFIX` — raw dataset prefix (default: `raw`)
- `S3_PROCESSED_PREFIX` — processed dataset prefix (default: `processed`)
- `S3_REFINED_PREFIX` — refined dataset prefix (default: `refined`)
- `S3_DASHBOARD_PREFIX` — dashboard dataset prefix (default: `processed/flights_dashboard`)
- `S3_MODEL_PREFIX` — model artifacts prefix (default: `models`)
- `S3_EXPLAIN_PREFIX` — SHAP outputs prefix (default: `explain`)
- `S3_PREDICTIONS_PREFIX` — weekly predictions prefix (default: `predictions`)
- `S3_ATHENA_RESULTS_PREFIX` — Athena query results prefix (default: `athena-results`)
- `S3_PREDICTIONS_ATHENA_PREFIX` — Athena dataset prefix for predictions (default: `<S3_PREDICTIONS_PREFIX>/weekly_predictions`)
- `S3_TABLE_LAYOUT` — S3 layout for Athena tables (`folder` recommended)
- `ATHENA_DATABASE` — Athena database name (default: `flight_advisor`)
- `ATHENA_TABLE` — table to sync partitions (default: `flights_processed`)
- `ATHENA_PREDICTIONS_TABLE` — Athena table for weekly predictions (default: `weekly_predictions`)
- `AWS_REGION` and `AWS_PROFILE` — AWS credentials resolution

### Optional: S3 + Athena pipeline

If you are running the pipeline on AWS (S3 + Athena), use the steps below.
All dataset interactions default to S3 when `S3_BUCKET` is set.

#### Step 1 — Upload raw datasets to S3 (CSV or Parquet)

```bash
python src/aws/uploader.py --bucket flight-advisor-fiap3 --input data/raw/flights.csv  --name flights.parquet
python src/aws/uploader.py --bucket flight-advisor-fiap3 --input data/raw/airports.csv --name airports.parquet
python src/aws/uploader.py --bucket flight-advisor-fiap3 --input data/raw/airlines.csv --name airlines.parquet
```
The uploader converts CSVs to Parquet and stores them in `s3://$S3_BUCKET/raw/`.

#### Step 2 — Preprocessing (reads from `raw/`, writes to `processed/` and `refined/`)

```bash
python src/preprocessing.py --bucket flight-advisor-fiap3 --athena-table-layout folder
```

For a first run or schema change, add `--athena-drop-existing`.
Preprocessing also registers Athena tables and syncs partitions unless you pass `--skip-athena`.
Use `--skip-partitioned` or `--skip-dashboard` if you want to disable specific outputs.

This step uploads the following to S3:
- `s3://$S3_BUCKET/processed/flights_processed.parquet`
- `s3://$S3_BUCKET/processed/train.parquet`
- `s3://$S3_BUCKET/processed/test.parquet`
- `s3://$S3_BUCKET/refined/airport_profiles.parquet`
- `s3://$S3_BUCKET/refined/flights_processed.parquet`
- `s3://$S3_BUCKET/refined/train.parquet`
- `s3://$S3_BUCKET/refined/test.parquet`
- `s3://$S3_BUCKET/processed/flights_processed/` (partitioned by `YEAR`/`MONTH` for Athena)
- `s3://$S3_BUCKET/processed/flights_dashboard/` (slim dataset for the dashboard)
- `s3://$S3_BUCKET/processed/train/` (Athena folder layout)
- `s3://$S3_BUCKET/processed/test/` (Athena folder layout)
- `s3://$S3_BUCKET/refined/airport_profiles/` (Athena folder layout)

#### Step 3 — Athena setup (folder layout)

```bash
# Normal execution
python src/aws/athena_client.py --bucket flight-advisor-fiap3 --format parquet --table-layout folder

# See the DDL without executing
python src/aws/athena_client.py --bucket flight-advisor-fiap3 --dry-run --format parquet --table-layout folder

# Recreate tables from scratch
python src/aws/athena_client.py --bucket flight-advisor-fiap3 --drop-existing --format parquet --table-layout folder
```

Notes:
- `flights_processed` is partitioned (`YEAR`, `MONTH`) and uses **folder layout**
  (e.g. `s3://$S3_BUCKET/processed/flights_processed/`).
- `train`, `test`, and `airport_profiles` also use **folder layout** for Athena.

Partition sync (preprocessing already runs this step):

```bash
# Athena Engine v3
CALL system.sync_partition_metadata('flight_advisor','flights_processed','ADD');
```

```bash
# Athena Engine v2 fallback
MSCK REPAIR TABLE flight_advisor.flights_processed;
```

#### Step 4 — (Optional) Rebuild only the dashboard dataset

```bash
python dashboard/build_dashboard_dataset.py \
  --input s3://$S3_BUCKET/processed/flights_processed/ \
  --bucket flight-advisor-fiap3 \
  --prefix processed/flights_dashboard \
  --overwrite
```

Legacy option: `dashboard/parquet_writer.py` can still build the partitioned dataset from a local
`data/flights_processed.parquet`, but preprocessing already produces the S3 partitions.

#### Step 5 — Weekly predictions published to Athena

```bash
# One command: generate future flights + predict
python src/jobs/weekly_pipeline.py --start-date 2026-03-23 --week-days 7 --rows 50000
```

This writes:
- `s3://$S3_BUCKET/$S3_PREDICTIONS_PREFIX/weekly_predictions_YYYYMMDD.parquet` (artifact file)
- `s3://$S3_BUCKET/$S3_PREDICTIONS_ATHENA_PREFIX/year=.../month=.../` (partitioned dataset)

And registers/syncs:
- `$ATHENA_DATABASE.$ATHENA_PREDICTIONS_TABLE` (Parquet, partitioned by `year`, `month`)

Notes:
- Athena publish runs by default when a bucket is available (`--bucket` / `S3_BUCKET`, or inferred from S3 input/output URIs).
- To disable publish for a run, use:

```bash
python src/jobs/weekly_pipeline.py --skip-athena-publish ...
```
#### AWS credentials diagnostics

```bash
# Verify which identity is being used
aws sts get-caller-identity

# Test Athena access
aws athena list-work-groups

# Test Glue access
aws glue get-database --name flight_advisor

# Configure new credentials
aws configure
```

### 4. Download and prepare the data

```bash
# Download from: https://www.kaggle.com/datasets/usdot/flight-delays
# Place the three files in data/raw/:
#   data/raw/flights.csv
#   data/raw/airports.csv
#   data/raw/airlines.csv

python src/preprocessing.py
```

### 5. Train the models

```bash
python src/trainer.py
# Default: reads s3://$S3_BUCKET/$S3_PROCESSED_PREFIX/train.parquet and test.parquet
# and uploads artifacts to s3://$S3_BUCKET/$S3_MODEL_PREFIX/
```

Local fallback (explicit CSV paths + no upload):

```bash
python src/trainer.py --train data/processed/train.parquet --test data/processed/test.parquet --no-upload
```

### 6. Generate SHAP explanations

```bash
python src/explainer.py --input s3://$S3_BUCKET/$S3_PROCESSED_PREFIX/test.parquet --plot
# Outputs: models/explain/shap_top_featured.csv and shap_summary.png
# Uploads: s3://$S3_BUCKET/$S3_EXPLAIN_PREFIX/
```

### 7. Create the vector store (RAG)

```bash
python src/rag/indexer.py
```

### 8. Start the API

```bash
uvicorn src.api.main:app --reload
# Access: http://localhost:8000/docs
```

### 9. Start the dashboard

```bash
python dashboard/app.py
```

---

## 📓 Notebooks

The notebooks are organized in order of execution and are self-contained — each includes context, code, and interpretation of the results.

| Notebook | Content | Highlights |
|---|---|---|
| `01_eda.ipynb` | Complete data exploration | Delay maps, distributions, seasonal patterns |
| `02_feature_engineering.ipynb` | Feature creation and validation | Time of day, holidays, target encoding of airports |
| `03_supervised_models.ipynb` | Comparison of 3 models + SHAP | ROC curves, SHAP summary plots, confusion matrix |
| `04_unsupervised_models.ipynb` | K-Means + PCA | Profile of each cluster, interactive 2D visualization |
| `05_anomaly_detection.ipynb` | Isolation Forest | Ranking of the most anomalous flights |

---

## 📊 Results

> *Results will be updated after full execution with the dataset.*

### Supervised Models

| Model | ROC-AUC | F1-Score | Accuracy |
|---|---|---|---|
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| Logistic Regression | — | — | — |

### Top Features (SHAP)

The most relevant features for delay prediction identified by the model will be documented here after training.

### Airport Clusters (K-Means)

| Cluster | Profile | Example Airports |
|---|---|---|
| 0 | — | — |
| 1 | — | — |
| 2 | — | — |
| 3 | — | — |

---

## 💼 Market Vision

Flight Advisor is conceived as a commercial product with three primary segments:

```
B2C  →  Passengers   →  Dashboard + reliability score per flight
B2B  →  Insurers     →  Delay risk API for dynamic pricing
B2B  →  Airlines     →  Cascading delay prediction for fleet management
```

**Addressable Market:**
- Travel insurance: **$23 billion** (global market)
- B2B travel tech: **$4.2 billion** (estimated SAM)
- Predictive flight tools: **$180 million** (direct niche)

For a complete market analysis, competition, and roadmap, consult the [project conception document](docs/flight_advisor_conception.docx).

---

## ⚠️ Limitations and Next Steps

### Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Historical dataset (not real-time) | Does not capture current weather or live ATC events | Documented in the interface; V1.0 to integrate live APIs |
| Imbalanced classes (~20% delays) | Risk of bias towards the majority class | `class_weight='balanced'` + SMOTE + F1/AUC metrics |
| Restricted geographic scope (USA) | Patterns may not generalize to other countries | Scope documented; expansion on the roadmap |
| Local embeddings (MiniLM) | Lower quality than API embeddings | Sufficient for MVP; upgrade planned |

### Next Steps

- [ ] Fine-tuning Qwen 3 with an aviation-specialized corpus
- [ ] Integration with FlightAware API for real-time data
- [ ] User authentication and query history
- [ ] White-label module for insurers
- [ ] Expansion to international flights

---

## 🗂️ Dataset

**Source:** [2015 Flight Delays and Cancellations — Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays)

The project uses **three CSV files** available in the dataset:

| File | Description | Key |
|---|---|---|
| `flights.csv` | Main base — 31 columns, ~5.8M records of domestic US flights | — |
| `airports.csv` | Geographic reference — name, city, state, lat/lon of each airport | `IATA_CODE` |
| `airlines.csv` | Airline reference — IATA code and full name | `IATA_CODE` |

The three files are related via foreign keys:

```
flights.csv ── AIRLINE             ──► airlines.csv (IATA_CODE)
            ── ORIGIN_AIRPORT      ──► airports.csv (IATA_CODE)
            ── DESTINATION_AIRPORT ──► airports.csv (IATA_CODE)
```

For the complete schema of all columns, cleaning rules, and feature engineering, consult [PROTOTYPE.md](PROTOTYPE.md).

> ⚠️ The raw files are not versioned in the repository due to their size. Download from the link above and place them in `data/raw/`.

### ☁️ Storage

Parquet reads and writes default to **AWS S3** when `S3_BUCKET` is set (raw CSVs are converted on upload). Model artifacts and SHAP outputs are automatically uploaded to S3 after training and explanation runs, keeping environments in sync.

---

## 👤 Author

<div align="center">

**Guilherme Lossio**

Machine Learning Engineering — FIAP MLET 2026

[![GitHub](https://img.shields.io/badge/GitHub-guilhermelossio-181717?style=flat-square&logo=github)](https://github.com/guilherme-lossio)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-guilhermelossio-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/guilherme-lossio)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-guilhermelossio-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/GuilhermeL)

</div>

---

<div align="center">

*Tech Challenge Phase 03 — FIAP Machine Learning Engineering*

**Flight Advisor** · Made with ☕ and lots of Python

</div>
