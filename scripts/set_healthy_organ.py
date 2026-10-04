"""Set the organ label of every image in a healthy-crop class to 'none'.

Healthy classes (disease_type == 'healthy' in data/class_attributes.csv, e.g. celery
leaf, corn leaf, coffee leaf) have no diseased plant part, so organ_review.final_label
becomes 'none', reviewed = TRUE, label_source = 'class_rule'. Missing rows are inserted.
This overrides earlier labels, human ones included (use --keep-human to spare them).

Run (dry run by default):
  DB_URL=... python scripts/set_healthy_organ.py [--apply] [--keep-human]
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write to the DB (default: dry run)")
    ap.add_argument("--keep-human", action="store_true", help="don't overwrite human-reviewed rows")
    args = ap.parse_args()

    attrs = pd.read_csv(ROOT / "data" / "class_attributes.csv", dtype=str).fillna("")
    healthy = set(attrs.loc[attrs["disease_type"] == "healthy", "class"])
    preds = pd.read_csv(ROOT / "data" / "organ_predictions.csv", dtype=str).fillna("")
    preds = preds[(preds.pred_organ != "") & preds["class"].isin(healthy)]
    print(f"{len(healthy)} healthy classes, {len(preds):,} images")

    url = os.environ.get("DB_URL") or sys.exit("Set DB_URL")
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    kwargs = {"pool_pre_ping": True}
    if url.startswith("postgresql+psycopg://"):  # psycopg3 + Supabase pooler: no prepared statements
        kwargs["connect_args"] = {"prepare_threshold": None}
    engine = create_engine(url, **kwargs)

    guard = "WHERE organ_review.label_source IS DISTINCT FROM 'human'" if args.keep_human else ""
    sql = text(f"""
        INSERT INTO organ_review (path, class, pred_organ, confidence, final_label, reviewed, reviewed_at, label_source)
        VALUES (:path, :cls, :pred_organ, :confidence, 'none', TRUE, :now, 'class_rule')
        ON CONFLICT (path) DO UPDATE SET final_label = 'none', reviewed = TRUE,
            reviewed_at = EXCLUDED.reviewed_at, label_source = 'class_rule' {guard}
    """)
    now = datetime.now(timezone.utc)
    rows = [{"path": r.path, "cls": r["class"], "pred_organ": r.pred_organ,
             "confidence": r.confidence, "now": now} for _, r in preds.iterrows()]
    with engine.connect() as c:
        done = c.execute(text("SELECT COUNT(*) FROM organ_review WHERE final_label = 'none'")).scalar()
        print(f"currently 'none' in DB: {done:,}")
        if not args.apply:
            print("dry run - re-run with --apply to set all of them")
            return
        for i in range(0, len(rows), 2000):
            c.execute(sql, rows[i:i + 2000])
        c.commit()
        print("now 'none':", c.execute(text("SELECT COUNT(*) FROM organ_review WHERE final_label = 'none'")).scalar())


if __name__ == "__main__":
    main()
