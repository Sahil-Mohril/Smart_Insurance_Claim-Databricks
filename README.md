# Smart Insurance Claim

An end-to-end Databricks Lakehouse solution for automated insurance claims processing. The platform ingests policy, customer, telematics, and claim-image data from multiple source systems, cleans and enriches it through a medallion (bronze/silver/gold) architecture, trains a computer-vision model to assess vehicle damage, and applies a configurable rule engine to flag claims for automatic fund release or further investigation. Results are served through DBSQL, AI/BI Dashboards, AI/BI Genie, and a Lakebase-backed serving layer.

## Architecture

<img width="718" height="317" alt="image" src="https://github.com/user-attachments/assets/9e3f64a9-bbae-419e-94e8-4a1dd7b7c596" />


| Layer | Description |
|---|---|
| **Source Systems** | Relational database (customers, policies), Kafka / Confluent streaming (vehicle telematics), and object store (claim images, accident-image training set) |
| **Ingestion** | [Lakeflow Connect](https://www.databricks.com/product/data-engineering/lakeflow-connect) for the relational database, [Lakeflow Declarative Pipelines](https://www.databricks.com/product/data-engineering/lakeflow-declarative-pipelines) with a streaming read against Kafka/Confluent for telematics events, and [Auto Loader](https://docs.databricks.com/ingestion/auto-loader) for images and metadata landing in cloud object storage |
| **Processing** | Bronze → Silver → Gold Lakeflow Declarative Pipelines that clean, validate, and join the datasets, plus an MLflow-tracked model training notebook and a SQL-based rule engine |
| **Consumption** | DBSQL, Lakebase, Model Serving (MosaicAI Serving), AI/BI Dashboards, AI/BI Genie, and Databricks Apps |

## Data Flow (Medallion Architecture)
<img width="910" height="428" alt="Screenshot 2026-09-13 175211" src="https://github.com/user-attachments/assets/2904d320-48f4-44f4-94e0-61219e2814c0" />


**Landing (`00_landing`)** — raw files/tables as delivered by each source:
- `telemetry`, `customers`, `policies`, `claims` (via Lakeflow Connect / Kafka-Confluent streaming ingestion)
- Claim images, claim-image metadata (CSV), and accident training images (via Auto Loader / Volumes)

**Bronze (`01_bronze`)** — raw data captured as-is into Delta tables:
- `telemetry`, `customer`, `policies`, `claims` — streamed in from landing tables
- `claim_images` — binary files read from a Unity Catalog Volume
- `claim_images_metadata` — CSV metadata picked up via Auto Loader (`cloudFiles`)
- `training_images` — accident training images ingested via Auto Loader (`cloudFiles`, `BINARYFILE`)

**Silver (`02_silver`)** — cleaned, validated, and enriched data with Lakeflow expectations (data-quality checks):
- `telemetry` — timestamp parsing, coordinate validation
- `policy` — premium normalization, non-null policy number check
- `claims` — date parsing, valid claim number/incident hour checks
- `customer` — name splitting/normalization, address construction, non-null customer ID check
- `training_images` — damage-severity label extracted from file path
- `claim_images` — image file name extracted from path
- `claims_rules` — table of configurable business rules (see Rule Engine below)
- `training_images_resized` — center-cropped/resized (224×224) JPEG images produced by the ML notebook

**Gold (`03_gold`)** — curated, business-ready tables:
- `aggregated_telematics` — average speed and location per vehicle chassis
- `customer_claim_policy` — claims joined with policy and customer records
- `customer_claim_policy_telematics` — the above enriched with geocoded customer address and aggregated telematics
- `claim_images_predicted` — damage-classification predictions from the vision model
- `customer_claim_policy_telematics_predicted` — claim/policy/telematics data joined with image predictions
- `claim_insights` — final output after all business rules are applied, indicating whether a claim should be auto-released or investigated further

## Machine Learning

`ML_notebook.py` fine-tunes a `microsoft/resnet-50` image classification model (via Hugging Face `transformers` + PyTorch) on the labeled accident/training images to classify vehicle damage severity (e.g., `ok`, `minor`, `major`). Training is tracked with MLflow, including the dataset, model signature, and a custom `pyfunc` wrapper for inference. The trained model is registered and served for scoring incoming claim images (`claim_images_predicted`).

## Rule Engine

`Rule_Engine.py` defines and stores claim-adjudication rules in `smart_claims_dev.02_silver.claims_rules`, then applies each active rule to `customer_claim_policy_telematics_predicted` to produce `claim_insights`. Rules include:

- **Valid policy date** — claim date falls within the policy's effective/expiry window
- **Valid claim amount** — claim value does not exceed the policy's insured sum
- **Severity match** — reported incident severity aligns with the model's predicted damage label
- **Speed check** — telematics speed at the time of the incident is within a normal range
- **Fund release** — if all prior checks pass, the claim is marked for automatic fund release; otherwise it's flagged for manual investigation

  Databricks AI/BI dashboard

  <img width="1917" height="862" alt="Screenshot 2026-09-13 174458" src="https://github.com/user-attachments/assets/d2f9e86a-55da-4632-8979-44759fdc28c0" />
  <img width="1912" height="870" alt="Screenshot 2026-09-13 174508" src="https://github.com/user-attachments/assets/d23590a7-0b81-43c9-846b-9236b7604b7f" />



## Repository Structure

```
Smart Insurance Claim/
├── Insurance_claim/
│   ├── requirements.txt                         # Python dependencies (geopy, pandas)
│   ├── ML_notebook.py                            # Image classification model training (MLflow)
│   ├── Rule_Engine.py                            # Claims business-rule definitions and evaluation
│   ├── testing.py                                # Ad-hoc validation queries
│   ├── Smart Claims Analysis.lvdash.json         # AI/BI dashboard definition
│   ├── smart_claims_pipline/
│   │   └── transformations/
│   │       ├── bronze_transformation.py          # Bronze: telemetry (Kafka/Confluent stream)
│   │       └── sql_server_transformation.py       # Bronze: customer, policy, claims (Lakeflow Connect)
│   ├── Object_ingestion_pipline/
│   │   └── transformations/
│   │       ├── claim_img_pipeline.py             # Bronze: claim images (Volume read)
│   │       ├── metadata_pipline.py               # Bronze: claim image metadata (Auto Loader)
│   │       └── training_imgs.py                  # Bronze: training images (Auto Loader)
│   └── transformations/
│       ├── silver.py                             # Silver: cleaning & data-quality expectations
│       └── gold.py                               # Gold: aggregation, joins, geocoding
└── manifest.mf                                    # Databricks project manifest
```

## Prerequisites

- A Databricks workspace with Unity Catalog enabled
- A Kafka / Confluent Cloud cluster (or topic) streaming vehicle telematics events
- Source relational database (customers, policies) accessible via Lakeflow Connect
- Cloud object storage (Volume) containing claim images, claim-image metadata, and accident training images
- Python dependencies listed in `Insurance_claim/requirements.txt`:
  - `geopy`
  - `pandas`

## Getting Started

1. **Provision the catalog/schemas** — create the `smart_claims_dev` catalog with `00_landing`, `01_bronze`, `02_silver`, and `03_gold` schemas in Unity Catalog.
2. **Configure ingestion**
   - Set up Lakeflow Connect for the relational source (customers, policies).
   - Configure the Kafka/Confluent connection (bootstrap servers, topic, and credentials) for telematics streaming ingestion.
   - Point Auto Loader at the object storage Volumes for claim images, image metadata, and training images.
3. **Run the Lakeflow Declarative Pipelines** to populate bronze, silver, and gold tables (`smart_claims_pipline`, `Object_ingestion_pipline`, and `transformations`).
4. **Train the damage-classification model** by running `ML_notebook.py`; review the run in MLflow and register the model.
5. **Score claim images** with the registered model to produce `claim_images_predicted`.
6. **Define/refresh business rules and generate insights** by running `Rule_Engine.py` to produce the `claim_insights` table.
7. **Explore the results** via the `Smart Claims Analysis.lvdash.json` AI/BI Dashboard, DBSQL, or AI/BI Genie.

## Consumption Layer

- **DBSQL** — ad hoc and BI queries against gold tables
- **AI/BI Dashboards** — `Smart Claims Analysis.lvdash.json` visualizes claim insights
- **AI/BI Genie** — natural-language Q&A over claims data
- **Lakebase / MosaicAI Serving** — low-latency model serving and operational data access for downstream Apps
