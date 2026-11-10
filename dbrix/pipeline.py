# Databricks notebook source
# MAGIC %md
# MAGIC # Raw → silver → gold
# MAGIC Coffee-shop orders. Raw keeps the messy extract. Silver cleans it. Gold sums completed orders.
# MAGIC
# MAGIC `createOrReplaceTempView` saves a table for this Spark session.
# MAGIC `spark.table` reads it back.
# MAGIC On Databricks, the lasting version of a save is:
# MAGIC `df.write.format("delta").mode("overwrite").saveAsTable("dbrix_raw.orders")`

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType
from pyspark.sql.window import Window

spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw
# MAGIC Land the extract. Every column stays a string. Nothing is cleaned.

# COMMAND ----------

raw_orders = spark.createDataFrame(
    [
        ("1001", "2026-01-03 09:15:00", "C01", "S1", "COMPLETED", "$12.50"),
        ("1001", "2026-01-03 09:16:00", "C01", "S1", "completed", "$12.50"),
        ("1002", "01/03/2026 10:02", "C02", "S1", " Completed ", "$8.00"),
        ("1003", "2026-01-03T11:45:00", "C03", "S2", "cancelled", "$15.00"),
        ("1004", "2026-01-04 08:00:00", "C01", "S2", "PENDING", "$4.25"),
        ("1005", None, "C04", "S1", "completed", "$9.00"),
        ("1006", "2026-01-04 14:20:00", None, "S1", "completed", "$6.00"),
        ("1007", "2026-01-05 09:00:00", "C02", "S3", "complet", "$20.00"),
        ("1008", "2026-01-05 16:30:00", "C05", "S1", "completed", "$0"),
        ("1009", "not-a-date", "C03", "S2", "completed", "$11.00"),
        ("1010", "2026-01-06 12:00:00", "C01", "S1", "completed", "$18.75"),
        ("1011", "2026-01-06 13:10:00", "C06", "S2", "canceled", "$7.50"),
        ("1012", "2026-01-07 10:00:00", "C02", "S1", "completed", "$22.00"),
        ("1013", "2026-01-07 11:11:00", "C04", "S3", "completed", "$5.50"),
        ("1014", "2026-01-08 09:45:00", "C05", "S2", "completed", "$14.00"),
        ("1015", "2026-01-08 15:00:00", "C03", "S1", "completed", "$3.25"),
        ("1016", "2026-01-09 12:30:00", "C06", "S2", "completed", "$1,250.00"),
        ("1017", "2026-01-09 18:00:00", "C05", "S1", "completed", "n/a"),
    ],
    "order_id string, order_ts string, customer_id string, store_id string, status string, amount string",
)
raw_orders.createOrReplaceTempView("raw_orders")

raw_customers = spark.createDataFrame(
    [
        ("C01", "  Ada Lovelace ", "Ada@Example.com", "London", "2025-11-01"),
        ("C02", "Grace Hopper", "grace@example.com", "London", "2025-12-12"),
        ("C03", "Alan Turing", "alan@example.com", "Manchester", "2025-10-03"),
        ("C04", "Katherine Johnson", "kj@example.com", "Manchester", "2026-01-02"),
        ("C05", "Margaret Hamilton", "mh@example.com", "Leeds", "2025-09-15"),
        ("C06", "Tim Berners-Lee", None, "London", "2025-08-20"),
        ("C01", "  Ada Lovelace ", "Ada@Example.com", "London", "2025-11-01"),
    ],
    "customer_id string, full_name string, email string, city string, signup_date string",
)
raw_customers.createOrReplaceTempView("raw_customers")

raw_stores = spark.createDataFrame(
    [
        ("S1", "King's Cross", "London"),
        ("S2", "Northern Quarter", "Manchester"),
        ("S3", "City Centre", "Leeds"),
    ],
    "store_id string, store_name string, city string",
)
raw_stores.createOrReplaceTempView("raw_stores")

print("raw orders")
spark.table("raw_orders").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver
# MAGIC Read `raw_orders`. Trim text, parse the timestamp and amount, map status,
# MAGIC and send bad rows to `silver_orders_rejected`. If an order id appears twice, keep the latest timestamp.

# COMMAND ----------

orders = spark.table("raw_orders")

for column in ["order_id", "order_ts", "customer_id", "store_id", "status", "amount"]:
    trimmed = F.trim(F.col(column))
    orders = orders.withColumn(column, F.when(trimmed == "", None).otherwise(trimmed))

orders = (
    orders.withColumn(
        "order_ts_parsed",
        F.coalesce(
            F.try_to_timestamp("order_ts", F.lit("yyyy-MM-dd HH:mm:ss")),
            F.try_to_timestamp("order_ts", F.lit("yyyy-MM-dd'T'HH:mm:ss")),
            F.try_to_timestamp("order_ts", F.lit("MM/dd/yyyy HH:mm")),
        ),
    )
    .withColumn(
        "amount_num",
        F.regexp_replace("amount", r"[$,\s]", "").try_cast(DecimalType(12, 2)),
    )
    .withColumn("status_norm", F.lower(F.col("status")))
)
orders = orders.withColumn(
    "status_clean",
    F.when(F.col("status_norm").isin("completed", "complete"), "completed")
    .when(F.col("status_norm").isin("cancelled", "canceled"), "cancelled")
    .when(F.col("status_norm") == "pending", "pending"),
)
orders = orders.withColumn(
    "reject_reason",
    F.when(F.col("order_id").isNull(), "missing_order_id")
    .when(F.col("customer_id").isNull(), "missing_customer_id")
    .when(F.col("store_id").isNull(), "missing_store_id")
    .when(F.col("order_ts_parsed").isNull(), "bad_timestamp")
    .when(F.col("amount_num").isNull(), "bad_amount")
    .when(F.col("status_clean").isNull(), "bad_status"),
)

bad_orders = orders.filter(F.col("reject_reason").isNotNull())
good_orders = orders.filter(F.col("reject_reason").isNull())

latest = Window.partitionBy("order_id").orderBy(F.col("order_ts_parsed").desc())
good_orders = good_orders.withColumn("_rn", F.row_number().over(latest))
duplicate_orders = good_orders.filter(F.col("_rn") > 1).withColumn(
    "reject_reason", F.lit("duplicate_order_id")
)

silver_orders = good_orders.filter(F.col("_rn") == 1).select(
    "order_id",
    F.col("order_ts_parsed").alias("order_ts"),
    F.to_date("order_ts_parsed").alias("order_date"),
    "customer_id",
    "store_id",
    F.col("status_clean").alias("status"),
    F.col("amount_num").alias("amount"),
)
silver_orders.createOrReplaceTempView("silver_orders")

silver_orders_rejected = bad_orders.unionByName(duplicate_orders.drop("_rn")).select(
    "order_id", "order_ts", "customer_id", "store_id", "status", "amount", "reject_reason"
)
silver_orders_rejected.createOrReplaceTempView("silver_orders_rejected")

print("rejected orders")
spark.table("silver_orders_rejected").orderBy("order_id").show(truncate=False)
print("silver orders")
spark.table("silver_orders").orderBy("order_id").show(truncate=False)

customers = spark.table("raw_customers")
for column in ["customer_id", "full_name", "email", "city", "signup_date"]:
    trimmed = F.trim(F.col(column))
    customers = customers.withColumn(column, F.when(trimmed == "", None).otherwise(trimmed))

silver_customers = (
    customers.withColumn("email", F.lower(F.col("email")))
    .withColumn("signup_date", F.try_to_date("signup_date"))
    .filter(F.col("customer_id").isNotNull() & F.col("full_name").isNotNull())
    .dropDuplicates(["customer_id"])
    .select("customer_id", "full_name", "email", "city", "signup_date")
)
silver_customers.createOrReplaceTempView("silver_customers")

stores = spark.table("raw_stores")
for column in ["store_id", "store_name", "city"]:
    trimmed = F.trim(F.col(column))
    stores = stores.withColumn(column, F.when(trimmed == "", None).otherwise(trimmed))

silver_stores = (
    stores.filter(F.col("store_id").isNotNull() & F.col("store_name").isNotNull())
    .dropDuplicates(["store_id"])
    .select("store_id", "store_name", "city")
)
silver_stores.createOrReplaceTempView("silver_stores")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold
# MAGIC Read the silver tables. Revenue is completed orders only.

# COMMAND ----------

orders = spark.table("silver_orders")
customers = spark.table("silver_customers")
stores = spark.table("silver_stores")

gold_daily_store_sales = (
    orders.filter(F.col("status") == "completed")
    .groupBy("order_date", "store_id")
    .agg(
        F.count("*").alias("orders"),
        F.sum("amount").alias("revenue"),
        F.round(F.avg("amount"), 2).alias("avg_order_value"),
    )
    .join(stores.select("store_id", "store_name", "city"), "store_id", "left")
    .select(
        "order_date",
        "store_id",
        "store_name",
        "city",
        "orders",
        "revenue",
        "avg_order_value",
    )
)
gold_daily_store_sales.createOrReplaceTempView("gold_daily_store_sales")

gold_customer_summary = (
    orders.groupBy("customer_id")
    .agg(
        F.count("*").alias("orders"),
        F.sum(F.when(F.col("status") == "completed", 1).otherwise(0)).alias("completed_orders"),
        F.coalesce(
            F.sum(F.when(F.col("status") == "completed", F.col("amount"))),
            F.lit(0).cast("decimal(12,2)"),
        ).alias("lifetime_revenue"),
        F.min("order_date").alias("first_order_date"),
        F.max("order_date").alias("last_order_date"),
    )
    .join(customers.select("customer_id", "full_name", "city"), "customer_id", "left")
    .select(
        "customer_id",
        "full_name",
        "city",
        "orders",
        "completed_orders",
        "lifetime_revenue",
        "first_order_date",
        "last_order_date",
    )
)
gold_customer_summary.createOrReplaceTempView("gold_customer_summary")

print("daily store sales")
spark.table("gold_daily_store_sales").orderBy("order_date", "store_id").show(truncate=False)
print("customer summary")
spark.table("gold_customer_summary").orderBy("customer_id").show(truncate=False)

total = spark.table("gold_daily_store_sales").select(F.sum("revenue")).collect()[0][0]
assert spark.table("raw_orders").count() == 18
assert spark.table("silver_orders").count() == 12
assert spark.table("silver_orders_rejected").count() == 6
assert abs(float(total) - 1334.00) < 0.001
print(f"done. completed revenue = {total}")
