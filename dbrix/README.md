# DBrix

One PySpark script. Coffee-shop orders go raw, then silver, then gold.

Read `pipeline.py` from top to bottom.

- `createOrReplaceTempView("raw_orders")` saves the table for this Spark session
- `spark.table("raw_orders")` reads it back
- Silver trims text, parses dates and amounts, and splits bad rows into `silver_orders_rejected`
- Gold sums **completed** orders into `gold_daily_store_sales` and `gold_customer_summary`

## Run locally

```powershell
cd C:\Users\MuhammadYahya\Documents\Repos\Extras\DBrix
.\.venv\Scripts\Activate.ps1
python run_local.py
```

You should see 18 raw orders, 12 silver orders, 6 rejected orders, and completed revenue **1334.00**.

## Run on Databricks

Import `pipeline.py`, attach a cluster on Runtime 14 or newer, and run it.

Temp views last for the session. To keep a table, replace a save line with:

```python
raw_orders.write.format("delta").mode("overwrite").saveAsTable("dbrix_raw.orders")
```

and read it back with `spark.table("dbrix_raw.orders")`.
