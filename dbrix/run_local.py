"""Start local Spark and run pipeline.py."""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline.py"


def pipeline_source() -> str:
    kept = []
    for line in PIPELINE.read_text(encoding="utf-8").splitlines():
        if (
            line.startswith("# Databricks notebook source")
            or line.startswith("# COMMAND")
            or line.startswith("# MAGIC")
        ):
            kept.append("")
        else:
            kept.append(line)
    return "\n".join(kept) + "\n"


def ensure_java() -> None:
    homes = []
    if os.environ.get("JAVA_HOME"):
        homes.append(Path(os.environ["JAVA_HOME"]))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        for root in (
            Path(local) / "Programs" / "Temurin17",
            Path(local) / "Programs" / "Eclipse Adoptium",
        ):
            if root.is_dir():
                homes.extend(path for path in root.iterdir() if path.is_dir())
    homes.sort(key=lambda path: (0 if path.name.startswith("jdk-17") else 1, path.name))
    for home in homes:
        if (home / "bin" / "java.exe").is_file():
            os.environ["JAVA_HOME"] = str(home)
            os.environ["PATH"] = str(home / "bin") + os.pathsep + os.environ.get("PATH", "")
            return
    if shutil.which("java") is None:
        sys.exit("Java was not found. Install Temurin 17, then run this script again.")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ensure_java()
    os.chdir(ROOT)

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("dbrix")
        .master("local[1]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        exec(compile(pipeline_source(), str(PIPELINE), "exec"), {"spark": spark, "__name__": "__main__"})
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
