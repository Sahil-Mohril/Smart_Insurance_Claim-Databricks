# Databricks notebook source
# MAGIC %sql
# MAGIC select count(*) from smart_claims_dev.00_landing.telemetry;

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS smart_claims_dev.01_bronze.image_metadata;

# COMMAND ----------

# MAGIC %sql
# MAGIC