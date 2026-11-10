"""Kainos-style cheat sheet. From this folder:

    .\\.venv\\Scripts\\Activate.ps1
    python interview_cheatsheet.py

Say the idea first, then the code. The sections below are the usual prompts:
Spark session, DataFrame vs SQL, windows, incremental merge, comprehensions
vs generators, and a password checker.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from run_local import ensure_java


def check_password(password: str) -> list[str]:
    """Return every failed rule. An empty list means the password is valid.

    Talk through this before typing:
    - ask which rules, and whether they want a bool or the reasons
    - reject None and blank first
    - test each character, not the whole string (isupper() on "Ab1!" is False)
    - do not print or log the password
    """
    if not isinstance(password, str) or password.strip() == "":
        return ["password is required"]

    specials = set("!@#$%^&*()-_=+[]{}|;:'\",.<>?/")
    checks = [
        (len(password) >= 8, "at least 8 characters"),
        (any(char.isupper() for char in password), "an uppercase letter"),
        (any(char.islower() for char in password), "a lowercase letter"),
        (any(char.isdigit() for char in password), "a digit"),
        (any(char in specials for char in password), "a special character"),
        (not any(char.isspace() for char in password), "no spaces"),
    ]
    return [message for passed, message in checks if not passed]


def python_one_liners() -> None:
    print("\n===== python: comprehension vs generator =====")
    nums = [1, 2, 3, 4]

    # Comprehension builds the whole list now.
    squares = [n * n for n in nums if n % 2 == 0]
    # Generator expression is lazy. Nothing runs until you iterate it.
    # any() / next() can stop early. A comprehension cannot.
    squares_gen = (n * n for n in nums if n % 2 == 0)
    print("list", squares, "generator", type(squares_gen).__name__, list(squares_gen))

    def squares_yield(values):
        for n in values:
            yield n * n

    print("yield", list(squares_yield(nums)))

    # Dict comprehension. Duplicate keys: the last one wins.
    by_id = {row["id"]: row["name"] for row in ({"id": 1, "name": "Ada"}, {"id": 1, "name": "Ada L"})}
    print("last key wins", by_id)

    # Mutable default is created once, then reused across calls. Use None.
    def add_bad(item, bucket=[]):
        bucket.append(item)
        return bucket

    def add_ok(item, bucket=None):
        bucket = [] if bucket is None else bucket
        bucket.append(item)
        return bucket

    print("shared default", add_bad(1), add_bad(2))
    print("fresh default", add_ok(1), add_ok(2))
    print("is None", None is None, "not a substitute for == on numbers")

    print("\n===== password checker =====")
    samples = ["", "short1!", "alllowercase1!", "NoSpecial1", "GoodPass1!"]
    for sample in samples:
        failed = check_password(sample)
        print(f"{sample!r:20} -> {failed or 'valid'}")
    assert check_password("GoodPass1!") == []
    assert "at least 8 characters" in check_password("Ab1!")
    assert "no spaces" in check_password("Good Pass1!")


def spark_cheatsheet(spark: SparkSession) -> None:
    print("\n===== create a dataframe, save it, read it =====")
    # A long value makes truncate obvious: show() cuts cells at 20 chars by default.
    people = spark.createDataFrame(
        [
            (1, "Ada Lovelace", "engineering", 90),
            (2, "Grace Hopper", "engineering", 90),
            (3, "Alan Turing", "research", 70),
            (3, "Alan Turing", "research", 70),
        ],
        "id int, name string, dept string, score int",
    )
    people.createOrReplaceTempView("people")
    spark.table("people").show(truncate=False)

    print("\n===== same filter: DataFrame API and SQL =====")
    # F is pyspark.sql.functions: SQL functions as Python calls.
    # spark.sql is the query as text. Catalyst compiles both to one plan.
    # Use SQL when the question is a query. Use F when you are building steps.
    via_api = people.filter(F.col("dept") == "engineering").select("name", "score")
    via_sql = spark.sql("SELECT name, score FROM people WHERE dept = 'engineering'")
    print("api", via_api.count(), "sql", via_sql.count())

    # F.col("score") is a column. F.lit(1) is a constant turned into a column.
    # A bare Python string inside try_to_timestamp is read as a column name, so the
    # format has to be F.lit("yyyy-MM-dd"). In F.when, a plain "yes" is accepted too.
    labelled = people.select(
        "name",
        F.when(F.col("score") >= 90, "high").otherwise("other").alias("band"),
        F.lit("interview").alias("source"),
    )
    labelled.show(truncate=False)

    print("\n===== window: row_number, rank, dense_rank, lag, running sum =====")
    # Two engineers both score 90, so rank ties and then skips. dense_rank does not skip.
    # row_number always gives 1, 2, 3. Add name to the order so the tie is stable.
    order = Window.partitionBy("dept").orderBy(F.col("score").desc(), F.col("name"))
    ranked = people.dropDuplicates(["id"]).select(
        "dept",
        "name",
        "score",
        F.row_number().over(order).alias("row_number"),
        F.rank().over(order).alias("rank"),
        F.dense_rank().over(order).alias("dense_rank"),
        F.lag("score").over(order).alias("previous_score"),
        F.sum("score").over(order.rowsBetween(Window.unboundedPreceding, Window.currentRow)).alias("running_sum"),
    )
    ranked.orderBy("dept", "row_number").show(truncate=False)

    spark.sql(
        """
        SELECT dept, name, score,
               row_number() OVER (PARTITION BY dept ORDER BY score DESC, name) AS row_number,
               rank()       OVER (PARTITION BY dept ORDER BY score DESC, name) AS rank,
               dense_rank() OVER (PARTITION BY dept ORDER BY score DESC, name) AS dense_rank
        FROM (SELECT DISTINCT id, dept, name, score FROM people)
        """
    ).orderBy("dept", "row_number").show(truncate=False)

    print("\n===== joins and the rows that are new =====")
    depts = spark.createDataFrame(
        [("engineering", "Belfast"), ("research", "London")],
        "dept string, office string",
    )
    people.dropDuplicates(["id"]).join(depts, "dept", "left").select("name", "office").show()

    incoming = spark.createDataFrame([(4, "Katherine Johnson")], "id int, name string")
    # left_anti: keys in the left frame that are not in the right frame.
    incoming.join(people, "id", "left_anti").show()

    print("\n===== incremental merge =====")
    # On Databricks this is one Delta statement. Rerunning the same batch is safe
    # because the match is the business key, not "append everything".
    merge_sql = """
    MERGE INTO target t
    USING updates u
    ON t.id = u.id
    WHEN MATCHED AND u.op = 'D' THEN DELETE
    WHEN MATCHED AND u.updated_at >= t.updated_at THEN UPDATE SET *
    WHEN NOT MATCHED AND u.op <> 'D' THEN INSERT *
    """
    print(merge_sql.strip())

    # Same upsert without Delta: union, then keep the newest row per id.
    existing = spark.createDataFrame(
        [(1, "Ada", 10, "2026-01-01"), (2, "Grace", 20, "2026-01-01")],
        "id int, name string, amount int, updated_at string",
    )
    updates = spark.createDataFrame(
        [(2, "Grace", 25, "2026-01-02"), (3, "Alan", 30, "2026-01-02")],
        "id int, name string, amount int, updated_at string",
    )
    newest = Window.partitionBy("id").orderBy(F.col("updated_at").desc())
    merged = (
        existing.unionByName(updates)
        .withColumn("_rn", F.row_number().over(newest))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .orderBy("id")
    )
    merged.show()
    assert merged.count() == 3
    assert merged.filter(F.col("id") == 2).collect()[0]["amount"] == 25

    print("\n===== actions vs transformations, in one breath =====")
    # filter / select / withColumn / join / groupBy build a plan. They do not run.
    # show, count, collect, write, and createOrReplaceTempView's first use run it.
    # collect() pulls every row to the driver. show() and take() do not.
    # groupBy, join, and window shuffle. select and filter usually do not.
    # cache() keeps the result after the first action. unpersist() when you are done.
    wide = people.groupBy("dept").agg(F.count("*").alias("people"), F.max("score").alias("top_score"))
    wide.cache()
    wide.show()
    wide.unpersist()


def main() -> None:
    python_one_liners()

    # A laptop needs JAVA_HOME. Databricks already has a session named spark.
    ensure_java()
    spark = (
        SparkSession.builder.appName("kainos-cheatsheet")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        spark_cheatsheet(spark)
    finally:
        spark.stop()
    print("\ncheat sheet finished")


if __name__ == "__main__":
    main()
