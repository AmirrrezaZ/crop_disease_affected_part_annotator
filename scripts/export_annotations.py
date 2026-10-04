"""Pull the human annotations out of the database into CSV files.

Annotations live in the Supabase Postgres database (DB_URL), in two tables:
  organ_review      one row per image: final_label (affected part), reviewed, reviewed_at, label_source
  attribute_review  one row per (image, attribute): color/texture/shape/pattern value
Rows a person touched have label_source = 'human' and reviewed = TRUE.

Run:
  DB_URL=... python scripts/export_annotations.py [--out exports] [--all]
Writes <out>/organ_review.csv, <out>/attribute_review.csv (long) and
<out>/attribute_review_wide.csv (one row per image, one column per attribute).
By default only human-reviewed rows are exported; --all exports everything.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exports")
    ap.add_argument("--all", action="store_true", help="include unreviewed / GPT / rule rows")
    args = ap.parse_args()

    url = os.environ.get("DB_URL") or sys.exit("Set DB_URL")
    if url.startswith("postgresql://"):  # use whichever driver is installed
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(url, pool_pre_ping=True, connect_args={"prepare_threshold": None})
    where = "" if args.all else "WHERE label_source = 'human'"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    organ = pd.read_sql(f"SELECT * FROM organ_review {where} ORDER BY path", engine)
    attrs = pd.read_sql(f"SELECT * FROM attribute_review {where} ORDER BY path, attribute", engine)
    organ.to_csv(out / "organ_review.csv", index=False)
    attrs.to_csv(out / "attribute_review.csv", index=False)
    wide = attrs.pivot(index=["path", "class"], columns="attribute", values="value").reset_index()
    wide.to_csv(out / "attribute_review_wide.csv", index=False)
    print(f"organ_review: {len(organ):,} rows, attribute_review: {len(attrs):,} rows -> {out}/")


if __name__ == "__main__":
    main()
