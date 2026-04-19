# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,imports
from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Row, Column

# COMMAND ----------

# DBTITLE 1,inputs
dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")

if env not in ["dev", "staging", "prod"]:
    raise ValueError(f"Invalid env: {env}")

# COMMAND ----------

# DBTITLE 1,variables
schema     = "bronze"
table      = "yellow_taxi_raw"
audit_tbl  = f"{env}_catalog.audit.yellow_taxi_cast_failures"
target_tbl = f"{env}_catalog.{schema}.{table}"

print("Environment : ", env)
print("Target Table: ", target_tbl)
print("Audit Table : ", audit_tbl)

# COMMAND ----------

# DBTITLE 1,Read raw files
RAW_PATH = "/databricks-datasets/nyctaxi/tripdata/yellow/"

raw_df = (
    spark.read
    .format("csv")
    .option("header", "true")
    .load(RAW_PATH)
)

# COMMAND ----------

raw_df.limit(100).display()

# COMMAND ----------

# DBTITLE 1,Target schema
from pyspark.sql.types import (
    IntegerType,
    TimestampType,
    DoubleType,
    StringType
)

TARGET_SCHEMA = {
    "vendor_id"         : StringType(),
    "pickup_datetime"   : TimestampType(),
    "dropoff_datetime"  : TimestampType(),
    "passenger_count"   : IntegerType(),
    "trip_distance"     : DoubleType(),
    "pickup_longitude"  : DoubleType(),
    "pickup_latitude"   : DoubleType(),
    "rate_code"         : IntegerType(),
    "store_and_fwd_flag": StringType(),
    "dropoff_longitude" : DoubleType(),
    "dropoff_latitude"  : DoubleType(),
    "payment_type"      : StringType(),
    "fare_amount"       : StringType(), 
    "surcharge"         : DoubleType(),
    "mta_tax"           : DoubleType(),  
    "tip_amount"        : DoubleType(),
    "tolls_amount"      : DoubleType(),
    "total_amount"      : DoubleType(),
}

# COMMAND ----------

# DBTITLE 1,Functions
def add_audit_columns(df: DataFrame) -> DataFrame:
    return (
        df
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_env",         F.lit(env))
    )

# COMMAND ----------

def make_cast_expr(c: str, target_type) -> tuple:
    """
    Pure function — takes a column name and target type,
    returns a tuple of (cast_expr, error_expr).
    """
    original_val = F.col(c).cast("string")
    try_cast     = F.col(c).try_cast(target_type).alias(c)

    error_entry  = F.when(
        F.col(c).isNotNull() & F.col(c).try_cast(target_type).isNull(),
        F.struct(
            F.lit(c).alias("col"),
            original_val.alias("original_value")
        )
    )

    return (try_cast, error_entry)


def build_errors_col(error_exprs: list) -> Column:
    """
    Folds a list of error expressions into a single array
    """
    return F.array_compact(F.array(*error_exprs)).alias("_errors")


def apply_schema(df: DataFrame) -> DataFrame:
    # Partition columns into known and unknown
    known   = [(c, TARGET_SCHEMA[c]) for c in df.columns if c in TARGET_SCHEMA]
    unknown = [c for c in df.columns if c not in TARGET_SCHEMA]

    if unknown:
        print(f"Columns not in TARGET_SCHEMA - passed through as-is: {unknown}")

    all_exprs = [make_cast_expr(c, t) for c, t in known]
    cast_exprs, error_exprs = zip(*all_exprs)

    passthrough = [F.col(c) for c in unknown]

    return (
        df
        .select(
            *passthrough,
            *cast_exprs,
            build_errors_col(list(error_exprs))
        )
    )

# COMMAND ----------

final_df = (
    raw_df
    .transform(apply_schema)
    .transform(add_audit_columns)
)

# COMMAND ----------

final_df.limit(100).display()

# COMMAND ----------

clean_df      = final_df.filter(F.size("_errors") == 0)
quarantine_df = final_df.filter(F.size("_errors") > 0)

(
    clean_df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(target_tbl)
)

print(f"Saved clean_df to {target_tbl}")

(
    quarantine_df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(audit_tbl)
)

print(f"Saved quarantine_df to {audit_tbl}")
