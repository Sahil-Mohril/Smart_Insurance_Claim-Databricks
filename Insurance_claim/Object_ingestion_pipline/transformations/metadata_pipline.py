from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.table(
    name="smart_claims_dev.01_bronze.claim_images_metadata",
    comment="Bronze image metadata from landing CSV"
)
def image_metadata():

    df = (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("cloudFiles.inferColumnTypes", "false")
        .load(
            "/Volumes/smart_claims_dev/00_landing/claims/metadata/"
        )
    )

    return (
        df
        .withColumn(
            "_ingestion_timestamp",
            F.current_timestamp()
        )
        .withColumn(
            "_source_file",
            F.col("_metadata.file_path")
        )
    )