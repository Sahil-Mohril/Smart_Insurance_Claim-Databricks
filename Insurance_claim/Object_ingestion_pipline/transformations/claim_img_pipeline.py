from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.table(
    name="smart_claims_dev.01_bronze.claim_images",
    comment="Bronze claim images loaded from Unity Catalog Volume"
)
def claim_images():

    df = (
        spark.read
        .format("binaryFile")
        .load("/Volumes/smart_claims_dev/00_landing/claims/images/")
    )

    return (
        df
        .select(
            F.col("path").alias("image_path"),
            F.col("modificationTime").alias("image_modification_time"),
            F.col("length").alias("image_size"),
            F.col("content").alias("image_content")
        )
        .withColumn(
            "_ingestion_timestamp",
            F.current_timestamp()
        )
    )