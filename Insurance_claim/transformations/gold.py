import pandas as pd
import random

from typing import Iterator

from pyspark import pipelines as dp
from pyspark.sql.functions import (
    col,
    pandas_udf,
    avg
)


catalog = "smart_claims_dev"
silver_schema = "02_silver"
gold_schema = "03_gold"

def geocode(geolocator, address):

    try:
        
        return pd.Series({
            "latitude": random.uniform(-90, 90),
            "longitude": random.uniform(-180, 180)
        })
        # location = geolocator.geocode(address)
        
        # if location:
        #     return pd.Series({
        #         "latitude": location.latitude,
        #         "longitude": location.longitude
        #     })

    except Exception as e:
        print(f"Error getting lat/long: {e}")

    return pd.Series({
        "latitude": None,
        "longitude": None
    })


@pandas_udf("latitude float, longitude float")
def get_lat_long(
    batch_iter: Iterator[pd.Series]
) -> Iterator[pd.DataFrame]:

    import geopy
    geolocator = geopy.Nominatim(
        user_agent="claim_lat_long",
        timeout=5,
        scheme="https"
    )

    for address in batch_iter:
        yield address.apply(
            lambda x: geocode(geolocator, x)
        )
@dp.table(
    name=f"{catalog}.{gold_schema}.aggregated_telematics",
    comment="Average telematics by chassis",
    table_properties={
        "quality": "gold"
    }
)
def telematics():

    return (
        spark.read.table(
            f"{catalog}.{silver_schema}.telemetry"
        )
        .groupBy("chassis_no")
        .agg(
            avg("speed").alias("telematics_speed"),
            avg("latitude").alias("telematics_latitude"),
            avg("longitude").alias("telematics_longitude")
        )
    )



@dp.table(
    name=f"{catalog}.{gold_schema}.customer_claim_policy",
    comment="Curated claim joined with policy and customer records",
    table_properties={
        "quality": "gold"
    }
)
def customer_claim_policy():

    policy = spark.read.table(
        f"{catalog}.{silver_schema}.policy"
    )

    claim = spark.read.table(
        f"{catalog}.{silver_schema}.claims"
    )

    customer = spark.read.table(
        f"{catalog}.{silver_schema}.customer"
    )

    claim_policy = claim.join(
        policy,
        on="policy_no",
        how="inner"
    )

    return claim_policy.join(
        customer,
        claim_policy.CUST_ID == customer.customer_id,
        how="inner"
    )


@dp.table(
    name=f"{catalog}.{gold_schema}.customer_claim_policy_telematics",
    comment="Claims enriched with customer, policy, telematics and geolocation",
    table_properties={
        "quality": "gold"
    }
)
def customer_claim_policy_telematics():

    telematics = spark.read.table(
        f"{catalog}.{gold_schema}.aggregated_telematics"
    )

    customer_claim_policy = (
        spark.read.table(
            f"{catalog}.{gold_schema}.customer_claim_policy"
        )
        .where(
            col("BOROUGH").isNotNull()
        )
    )

    return (
        customer_claim_policy

        # Generate latitude/longitude from address
        .withColumn(
            "lat_long",
            get_lat_long(col("address"))
        )

        # Join telematics using chassis number
        .join(
            telematics,
            on="chassis_no",
            how="left"
        )
    )