# Spark and Python interview cheat sheet

Notes for a data engineering interview. Say the idea first, then write the code.

The runnable version is `interview_cheatsheet.py`.

```powershell
.\.venv\Scripts\Activate.ps1
python interview_cheatsheet.py
```

---

# Why these Spark calls

## `createOrReplaceTempView`

```python
raw_orders.createOrReplaceTempView("raw_orders")
spark.table("raw_orders")
```

This gives the DataFrame a name in the **current Spark session**. The next cell can read it with `spark.table` or `spark.sql`. Nothing is written to disk. The name disappears when the session stops. Calling it again replaces that name.

On Databricks, the version that **survives the session** is:

```python
raw_orders.write.format("delta").mode("overwrite").saveAsTable("dbrix_raw.orders")
spark.table("dbrix_raw.orders")
```

## `pyspark.sql.functions` vs a SQL query

```python
from pyspark.sql import functions as F

via_api = people.filter(F.col("dept") == "engineering").select("name", "score")
via_sql = spark.sql("SELECT name, score FROM people WHERE dept = 'engineering'")
```

`F` is the DataFrame API: one transformation per line. `spark.sql` is the same work written as a query. Both go through the Catalyst planner and become the same kind of job.

Use **SQL** when the question is already a query. Use **`F`** when you are building clean steps (trim, parse, flag, filter).

## `F.lit`

```python
F.lit("interview")
F.lit("yyyy-MM-dd HH:mm:ss")
```

`F.col("score")` means "the value in this row's score column". `F.lit` means "this constant, on every row".

Spark expressions only combine columns. A Python string is not a column until you wrap it. This matters for `try_to_timestamp`: a bare string is treated as a **column name**, so the date format has to be `F.lit("yyyy-MM-dd HH:mm:ss")`.

## `truncate=False`

```python
df.show(truncate=False)
```

`show()` cuts each cell at 20 characters by default. `truncate=False` prints the full value, so `Northern Quarter` and full timestamps stay readable.

---

# Start and stop Spark

```python
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("cheatsheet")
    .master("local[1]")  # local[*] uses every core; [1] is enough for a sample
    .config("spark.sql.shuffle.partitions", "1")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

try:
    spark.range(3).show()
finally:
    spark.stop()
```

| Call | What to say |
| --- | --- |
| `builder` | Configures the session before it starts |
| `getOrCreate()` | Reuses a live session in this process if one already exists |
| `spark.stop()` | Releases it. In a Databricks notebook you usually do **not** stop `spark` |

A laptop needs Java on `PATH` or `JAVA_HOME`. A Databricks cluster already has a session named `spark`.

## `try` / `spark.range(3).show()`

```python
try:
    spark.range(3).show()
finally:
    spark.stop()
```

`spark.range(3)` builds a one-column DataFrame called `id` with the rows `0`, `1`, and `2`. `show()` is the action that prints them. It is a smoke test: if this prints, the session started.

`try` / `finally` means `spark.stop()` still runs when the body throws. `finally` is not an error handler. `except` is what catches the error.

---

# Create a DataFrame, save it, read it

```python
people = spark.createDataFrame(
    [
        (1, "Ada Lovelace", "engineering", 90),
        (2, "Grace Hopper", "engineering", 90),
        (3, "Alan Turing", "research", 70),
    ],
    "id int, name string, dept string, score int",
)

people.createOrReplaceTempView("people")   # save
spark.table("people")                      # read
people.show(truncate=False)
```

Passing the schema as a DDL string keeps types obvious. `inferSchema` on a big CSV scans the file. Prefer an explicit schema.

## Schema from a list or a pandas Series

Spark will not invent a trustworthy schema from a bare Python list or a pandas Series. Pass the schema yourself.

From a list of tuples, the second argument is the schema. A DDL string or a `StructType` both work:

```python
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

rows = [(1, "Ada"), (2, "Grace")]

df = spark.createDataFrame(rows, "id int, name string")

schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("name", StringType(), True),
])
df = spark.createDataFrame(rows, schema)
```

The third `StructField` argument is nullable. `False` means the column is required.

A pandas Series is one column. Turn it into a frame, then pass the schema. Without that, Spark infers the type from the values.

```python
import pandas as pd

amount = pd.Series([10, 20, 30], name="amount")
df = spark.createDataFrame(amount.to_frame(), schema="amount long")
```

A Series has no room for a second column. For `id` and `name`, start from a pandas DataFrame or from tuples, not from one Series.

---

# Window functions

Two people in engineering both score 90. That tie is the whole question.

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

order = Window.partitionBy("dept").orderBy(F.col("score").desc(), F.col("name"))

ranked = people.select(
    "dept",
    "name",
    "score",
    F.row_number().over(order).alias("row_number"),
    F.rank().over(order).alias("rank"),
    F.dense_rank().over(order).alias("dense_rank"),
    F.lag("score").over(order).alias("previous_score"),
    F.sum("score").over(
        order.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    ).alias("running_sum"),
)
```

The same query in SQL:

```sql
SELECT dept, name, score,
       row_number() OVER (PARTITION BY dept ORDER BY score DESC, name) AS row_number,
       rank()       OVER (PARTITION BY dept ORDER BY score DESC, name) AS rank,
       dense_rank() OVER (PARTITION BY dept ORDER BY score DESC, name) AS dense_rank,
       lag(score)   OVER (PARTITION BY dept ORDER BY score DESC, name) AS previous_score
FROM people
```

| Function | Engineering scores 90, 90, then research 70 | What to say |
| --- | --- | --- |
| `row_number` | 1, 2 and 1 | Unique. Break ties with a second sort column |
| `rank` | 1, 1, then the next group starts at 1. Inside a tie the next rank **skips** | Tie, then jump (1, 1, 3) |
| `dense_rank` | 1, 1, 2 | Tie, then no jump |
| `lag` | null, then the previous score | Previous row in the window order |
| `sum` + `rowsBetween` | 90, then 180 | Running total, one row at a time |

`partitionBy` is the group. `orderBy` is the sequence inside the group. A window **shuffles**.

`dropDuplicates(["id"])` before the window if the source has repeated keys. Otherwise the ranks count the duplicates.

---

# Joins

```python
depts = spark.createDataFrame(
    [("engineering", "Belfast"), ("research", "London")],
    "dept string, office string",
)

people.join(depts, "dept", "left").select("name", "office")
```

`left` keeps every person. A missing department becomes null office.

```python
incoming = spark.createDataFrame([(4, "Katherine Johnson")], "id int, name string")
incoming.join(people, "id", "left_anti")
```

`left_anti` returns left rows whose key is **not** on the right. That is "what is new in this batch?"

---

# Incremental merge

Do not append the batch onto the target. Append duplicates the keys you have already loaded.

On Databricks, Delta does it in one statement. Replaying the same batch is safe because the match is the business key.

```sql
MERGE INTO target t
USING updates u
ON t.id = u.id
WHEN MATCHED AND u.op = 'D' THEN DELETE
WHEN MATCHED AND u.updated_at >= t.updated_at THEN UPDATE SET *
WHEN NOT MATCHED AND u.op <> 'D' THEN INSERT *
```

| Clause | Meaning |
| --- | --- |
| `ON t.id = u.id` | Business key |
| `op = 'D'` | Delete when the change feed says delete |
| `updated_at >=` | Ignore a late or older update |
| `WHEN NOT MATCHED` | Insert keys the target does not have yet |

Without Delta, union both sides and keep the newest row per id:

```python
newest = Window.partitionBy("id").orderBy(F.col("updated_at").desc())

merged = (
    existing.unionByName(updates)
    .withColumn("_rn", F.row_number().over(newest))
    .filter(F.col("_rn") == 1)
    .drop("_rn")
)
```

Example: Ada stays at 10, Grace moves from 20 to 25, Alan is inserted at 30.

If they ask for **history** (SCD2), do not overwrite. Close the open row (`is_current = false`, set `end_date`) and insert the new version.

---

# Transformations and actions

| Kind | Examples | What to say |
| --- | --- | --- |
| Transformation | `filter`, `select`, `withColumn`, `join`, `groupBy` | Builds a plan. Does not run |
| Action | `show`, `count`, `collect`, `write` | Runs the plan |
| Narrow | `select`, `filter` | Usually stays on the same partition. No shuffle |
| Wide | `groupBy`, `join`, window, `repartition` | Shuffle. This is the expensive part |

* `collect()` pulls **every** row to the driver. Use `show()` or `take(n)` to look.
* `cache()` keeps the result after the first action. Call `unpersist()` when you are done.
* `where` is the same method as `filter`.

```python
summary = (
    people.groupBy("dept")
    .agg(F.count("*").alias("people"), F.max("score").alias("top_score"))
)
summary.cache()
summary.show()
summary.unpersist()
```

---

# Comprehensions vs generators

```python
nums = [1, 2, 3, 4]

squares = [n * n for n in nums if n % 2 == 0]       # list, built immediately
squares_gen = (n * n for n in nums if n % 2 == 0)   # generator, lazy
```

| | Comprehension `[...]` | Generator `(...)` or `yield` |
| --- | --- | --- |
| When it runs | Immediately | When you iterate |
| Memory | Holds every result | One value at a time |
| Reuse | Read it again | Exhausted after one pass |
| Early exit | Builds the whole list first | `any()` and `next()` can stop |

```python
def squares_yield(values):
    for n in values:
        yield n * n
```

A **dict comprehension** keeps the last value when keys collide:

```python
by_id = {row["id"]: row["name"] for row in ({"id": 1, "name": "Ada"}, {"id": 1, "name": "Ada L"})}
# {1: "Ada L"}
```

---

# Mutable default argument

```python
def add_bad(item, bucket=[]):
    bucket.append(item)
    return bucket

add_bad(1)  # [1]
add_bad(2)  # [1, 2]  — same list as the first call
```

The default list is created **once**, when the function is defined, then reused. Fix it with `None`:

```python
def add_ok(item, bucket=None):
    bucket = [] if bucket is None else bucket
    bucket.append(item)
    return bucket
```

`None` check: use `x is None`, not `x == None`.

---

# Password checker

This is a common live-coding task. Talk, then type.

**Say this first**

1. Ask for the rules, and whether they want a bool or the reasons.
2. Reject a non-string and a blank password before the other rules.
3. Test **characters**, not the whole string. `"Ab1!".isupper()` is false, which is the wrong test.
4. Do not print or log the password.

**Rules used here:** length 8–64, no whitespace, one lowercase, one uppercase, one digit, one special.

```python
SPECIAL = set("!@#$%^&*()-_=+[]{};:,.<>?")

def password_failures(password: str) -> list[str]:
    if not isinstance(password, str):
        return ["must be a string"]
    failures = []
    if not 8 <= len(password) <= 64:
        failures.append("length must be 8-64")
    if any(c.isspace() for c in password):
        failures.append("no whitespace")
    if not any(c.islower() for c in password):
        failures.append("needs a lowercase letter")
    if not any(c.isupper() for c in password):
        failures.append("needs an uppercase letter")
    if not any(c.isdigit() for c in password):
        failures.append("needs a digit")
    if not any(c in SPECIAL for c in password):
        failures.append("needs a special character")
    return failures

def is_valid(password: str) -> bool:
    return password_failures(password) == []
```

| Password | Result |
| --- | --- |
| `""` | length, plus the missing character classes |
| `"short1!"` | length must be 8–64 |
| `"alllowercase1!"` | needs an uppercase letter |
| `"NoSpecial1"` | needs a special character |
| `"GoodPass1!"` | valid (`[]`) |

`any(...)` walks a generator and **stops at the first match**. That is the generator point, used on purpose inside the checker.

---

# One-liners worth saying out loud

| Question | Answer |
| --- | --- |
| Where vs having | `WHERE` filters rows before the group. `HAVING` filters after `GROUP BY` |
| Shuffle | Wide ops move rows across partitions so the same key lands together |
| `collect` | Action. Entire result comes to the driver. Dangerous on big data |
| `cache` | Reuse a DataFrame that you will action more than once |
| `left_anti` | Left rows with no match. The "new keys" join |
| Business key | The id you merge on. Not the file name, not the load timestamp |
| SCD2 | Keep history: close the old row, insert the new one |
| Temp view vs table | Temp view dies with the session. `saveAsTable` keeps a Delta table |

---

# More SQL questions

## `UNION` or `UNION ALL`?

`UNION` stacks rows and removes duplicates. `UNION ALL` stacks rows and keeps duplicates. `UNION ALL` is cheaper. Use it unless you actually need distinct rows.

## Why does `WHERE col = NULL` return nothing?

`NULL` means unknown. `unknown = unknown` is not true. Use `col IS NULL`. In Spark SQL, `col <=> NULL` is the null-safe equality.

## `COUNT(*)` or `COUNT(col)`?

`COUNT(*)` counts rows. `COUNT(col)` counts rows where `col` is not null. `COUNT(DISTINCT col)` counts different non-null values.

## `INNER JOIN` or `LEFT JOIN`?

`INNER` keeps only matching keys. `LEFT` keeps every row from the left table and fills the right side with null when there is no match.

## `WHERE` or `HAVING`?

`WHERE` filters raw rows, before grouping. `HAVING` filters groups, after `GROUP BY`.

```sql
SELECT dept, COUNT(*) AS n
FROM people
WHERE score >= 70
GROUP BY dept
HAVING COUNT(*) > 1
```

## What must appear in `GROUP BY`?

Every selected column that is not inside an aggregate (`COUNT`, `SUM`, `MAX`, …).

## `DELETE`, `TRUNCATE`, or `DROP`?

| Statement | What remains |
| --- | --- |
| `DELETE` | The table. Some or all rows are gone. Can have a `WHERE` |
| `TRUNCATE` | The table, with no rows |
| `DROP` | Nothing. The table itself is gone |

## `EXISTS` or `IN`?

Both answer "is this key in that set?". `EXISTS` stops at the first match and handles nulls more safely. `NOT IN` with a null in the list can wipe out the whole result.

## What is a CTE?

A named subquery you can read from the top:

```sql
WITH high AS (
  SELECT * FROM people WHERE score >= 90
)
SELECT dept, COUNT(*) FROM high GROUP BY dept
```

## What is a primary key?

The column, or columns, that identify one row. It is unique and not null. An index is separate: it speeds up lookup. A primary key is a rule about the data.

---

# More Python questions

## List or tuple?

A list is mutable: you can append and change items. A tuple is immutable. A tuple can be a dict key. A list cannot.

## `is` or `==`?

`==` asks whether the values are equal. `is` asks whether they are the same object. Use `is` for `None`. Use `==` for numbers and strings.

## Why can `""`, `[]`, `0`, and `None` all fail an `if`?

They are falsy. Everything else is mostly truthy, including `"0"` and `[0]`.

## `*args` and `**kwargs`?

`*args` collects extra positional arguments into a tuple. `**kwargs` collects extra named arguments into a dict.

```python
def log(message, *args, **kwargs):
    return message, args, kwargs
```

## `sort` or `sorted`?

`names.sort()` changes the list and returns `None`. `sorted(names)` returns a new list and leaves the original alone.

## Why `"".join(parts)` instead of `+=` in a loop?

Each `+=` on a string builds a new string. `join` builds one string at the end.

## Shallow copy or deep copy?

A shallow copy shares the inner lists. A deep copy duplicates them.

```python
import copy
inner = [[1]]
shallow = copy.copy(inner)
deep = copy.deepcopy(inner)
```

Changing `shallow[0]` also changes `inner[0]`. Changing `deep[0]` does not.

## What does a decorator do?

It wraps a function. `@cache` on `f` means `f = cache(f)`.

## `Exception`, `else`, `finally`?

`except` runs when that error is raised. `else` runs when the `try` did **not** raise. `finally` runs either way.

## Why not use a list as a dict key or a set item?

Dict keys and set items must be hashable and must not change. Lists and dicts can change, so they are unhashable. Tuples of immutable values are fine.

## `range` or a list of indexes?

```python
for i, name in enumerate(names):
    ...
```

`enumerate` gives the index and the value. You rarely need `range(len(names))`.
