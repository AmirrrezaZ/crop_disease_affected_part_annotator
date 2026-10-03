"""Seed the attribute_review table (color / texture / shape / pattern) from
data/class_attributes.csv.

Rules (applied per image, per attribute):
  * healthy class                              -> value 'none', reviewed
  * diseased class, attribute has no value     -> value 'none', reviewed
  * diseased class, attribute has ONE value    -> that value for every image of the
                                                  class, reviewed
  * diseased class, attribute has 2+ values    -> left unreviewed with an empty
                                                  value; a human (or the GPT pass)
                                                  picks which of the class's values
                                                  are visible in each image
label_source is 'class_rule' for the first three. Safe to re-run: rule rows only
overwrite rows that are still 'class_rule' or unreviewed, so human reviews and GPT
suggestions are never clobbered; multi-valued rows are insert-if-missing.

Run:
  DB_URL=... python scripts/seed_attributes.py [--dry-run]
(DB_URL is read from .env when python-dotenv is installed.)
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATTRS = ["color", "texture", "shape", "pattern"]
BATCH = 2000


def split(s):
    return [v for v in s.split(";") if v]


def build_rows():
    attrs = pd.read_csv(ROOT / "data" / "class_attributes.csv", dtype=str).fillna("")
    by_class = {r["class"]: r for _, r in attrs.iterrows()}
    preds = pd.read_csv(ROOT / "data" / "organ_predictions.csv", dtype=str).fillna("")
    preds = preds[preds.pred_organ != ""]

    rule, pending = [], []
    for path, cls in zip(preds.path, preds["class"]):
        r = by_class.get(cls)
        if r is None:
            continue
        for a in ATTRS:
            vals = split(r[a])
            if r["disease_type"] == "healthy" or not vals:
                rule.append((path, a, cls, "none"))
            elif len(vals) == 1:
                rule.append((path, a, cls, vals[0]))
            else:
                pending.append((path, a, cls, ""))
    return rule, pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    rule, pending = build_rows()
    per_attr_rule = Counter(a for _, a, _, _ in rule)
    per_attr_pending = Counter(a for _, a, _, _ in pending)
    print(f"{'attribute':10s}{'auto-reviewed':>15s}{'needs annotation':>18s}")
    for a in ATTRS:
        print(f"{a:10s}{per_attr_rule[a]:15,d}{per_attr_pending[a]:18,d}")
    if args.dry_run:
        return

    db_url = os.environ.get("DB_URL") or sys.exit("Set DB_URL")
    import psycopg2
    from psycopg2.extras import execute_values

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(open(ROOT / "scripts" / "schema.sql").read())
        conn.commit()

        for i in range(0, len(rule), BATCH):
            with conn.cursor() as cur:
                execute_values(cur, """
                    insert into attribute_review (path, attribute, class, value, reviewed, reviewed_at, label_source)
                    values %s
                    on conflict (path, attribute) do update set
                        class = excluded.class, value = excluded.value, reviewed = true,
                        reviewed_at = now(), label_source = 'class_rule'
                    where attribute_review.label_source = 'class_rule'
                       or attribute_review.reviewed = false
                """, rule[i:i + BATCH], template="(%s,%s,%s,%s,true,now(),'class_rule')")
            conn.commit()
        print(f"rule rows upserted: {len(rule):,}")

        for i in range(0, len(pending), BATCH):
            with conn.cursor() as cur:
                execute_values(cur, """
                    insert into attribute_review (path, attribute, class, value, reviewed)
                    values %s on conflict (path, attribute) do nothing
                """, pending[i:i + BATCH], template="(%s,%s,%s,%s,false)")
            conn.commit()
        print(f"pending rows inserted (if missing): {len(pending):,}")

        with conn.cursor() as cur:
            cur.execute("""select attribute, count(*), count(*) filter (where reviewed)
                           from attribute_review group by 1 order by 1""")
            for a, n, done in cur.fetchall():
                print(f"  {a:8s} rows={n:,} reviewed={done:,} ({done / n:.1%})")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
