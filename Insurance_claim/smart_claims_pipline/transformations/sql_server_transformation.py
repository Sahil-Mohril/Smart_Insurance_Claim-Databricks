from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame

@dp.table(
    name="smart_claims_dev.01_bronze.customer",
    comment="customer data"
)
def bronze_customer()->DataFrame:
    df = spark.readStream.table(
        "smart_claims_dev.00_landing.customers"
    )
    return df
@dp.table(
    name="smart_claims_dev.01_bronze.policies",
    comment="policy data"
)
def bronze_customer()->DataFrame:
    df = spark.readStream.table(
        "smart_claims_dev.00_landing.policies"
    )
    return df
@dp.table(
    name="smart_claims_dev.01_bronze.claims",
    comment="claim data"
)
def bronze_customer()->DataFrame:
    df = spark.readStream.table(
        "smart_claims_dev.00_landing.claims"
    )
    return df