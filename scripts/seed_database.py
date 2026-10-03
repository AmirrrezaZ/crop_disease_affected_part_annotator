"""One-time (and safely re-runnable) seed of the Supabase organ_review table:
every image in data/organ_predictions.csv becomes a row, defaulted to
unreviewed with final_label = the model's own prediction -- except where a
local organ_review.csv from the original crop_disease/version1 project has
a real human review already, which takes priority so existing work isn't
lost.

Requires:
  DB_URL   -- Supabase Postgres connection string (run scripts/schema.sql first)
Optional:
  LOCAL_REVIEW_CSV -- path to the existing organ_review.csv to import human
                      corrections from (default: sibling crop_disease/version1
                      checkout on this machine)

Run:
  DB_URL="postgresql://..." python scripts/seed_database.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data" / "organ_predictions.csv"
DEFAULT_LOCAL_REVIEW = (
    ROOT.parent / "crop_disease" / "version1" / "data" / "organ_review.csv")


def main():
    db_url = os.environ.get("DB_URL")
    if not db_url:
        sys.exit("Set DB_URL to your Supabase Postgres connection string first.")

    preds = pd.read_csv(PREDICTIONS, dtype=str).fillna("")
    preds = preds[preds.pred_organ != ""]  # skip unreadable images

    local_review_path = Path(os.environ.get("LOCAL_REVIEW_CSV", DEFAULT_LOCAL_REVIEW))
    reviewed_lookup = {}
    if local_review_path.exists():
        local = pd.read_csv(local_review_path, dtype=str).fillna("")
        local = local[local.reviewed == "True"]
        reviewed_lookup = {row["path"]: row for _, row in local.iterrows()}
        print(f"found {len(reviewed_lookup)} existing human reviews at {local_review_path}")
    else:
        print(f"no local review file at {local_review_path}; seeding all-unreviewed")

    rows = []
    for _, r in preds.iterrows():
        prior = reviewed_lookup.get(r["path"])
        if prior is not None:
            rows.append({"path": r["path"], "class": r["class"], "pred_organ": r.pred_organ,
                        "confidence": r.confidence, "final_label": prior["final_label"],
                        "reviewed": True, "reviewed_at": prior["reviewed_at"] or None})
        else:
            rows.append({"path": r["path"], "class": r["class"], "pred_organ": r.pred_organ,
                        "confidence": r.confidence, "final_label": r.pred_affected_part,
                        "reviewed": False, "reviewed_at": None})

    # Raw psycopg2 + execute_values for true multi-row round trips (SQLAlchemy's
    # default executemany over a list of dicts issues one round trip PER ROW,
    # which against a pooled connection is minutes-to-hours slower for 22k+ rows).
    # Commit per batch, not one giant transaction, so progress is visible and
    # a Ctrl-C / crash mid-run doesn't lose everything already upserted.
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                create table if not exists organ_review (
                    path text primary key, class text not null, pred_organ text,
                    confidence text, final_label text, reviewed boolean not null default false,
                    reviewed_at timestamptz, label_source text
                )
            """)
        conn.commit()

        batch = 2000
        cols = ["path", "class", "pred_organ", "confidence", "final_label", "reviewed", "reviewed_at"]
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            values = [tuple(r[c] for c in cols) for r in chunk]
            with conn.cursor() as cur:
                execute_values(cur, """
                    insert into organ_review (path, class, pred_organ, confidence, final_label, reviewed, reviewed_at)
                    values %s
                    on conflict (path) do update set
                        final_label = excluded.final_label,
                        reviewed = excluded.reviewed,
                        reviewed_at = excluded.reviewed_at
                """, values)
            conn.commit()
            print(f"upserted {min(i + batch, len(rows))}/{len(rows)}", flush=True)
    finally:
        conn.close()

    n_reviewed = sum(1 for r in rows if r["reviewed"])
    print(f"done: {len(rows)} rows total, {n_reviewed} carried over as already-reviewed")


if __name__ == "__main__":
    main()
