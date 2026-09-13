from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame

@dp.table(
    name="smart_claims_dev.01_bronze.telemetry",
    comment="Raw telemetry data from sensor",
)
def bronze()->DataFrame:

    df = spark.readStream.table(
        "smart_claims_dev.00_landing.telemetry"
    )
    return df