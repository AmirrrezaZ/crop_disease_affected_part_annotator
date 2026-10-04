"""Mark confident, still-unreviewed predictions of one affected part as reviewed.

Default: every image whose effective label is exactly 'leaf' (single part), whose CLIP
organ prediction is also leaf with confidence > 0.7, that nobody has reviewed yet, and
that is not in a healthy class (those are 'none'). They get final_label = leaf,
reviewed = TRUE, label_source = 'auto_conf' (so exports can tell them from human work).
Human-reviewed rows are never touched.

Effective label priority matches the app: per-image GPT > per-class GPT > CLIP.

Run (dry run by default):
  DB_URL=... python scripts/auto_confirm_part.py [--part leaf] [--threshold 0.7] [--apply]
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

LEGACY = {"seed": "seed_grain", "grain": "seed_grain", "seed/grain": "seed_grain",
          "whole_plant": "whole_plant_unknown", "not_plant_other": "whole_plant_unknown",
          "unknown": "whole_plant_unknown"}


def norm(v):
    v = str(v).strip().lower()
    return LEGACY.get(v, v)


def candidates(part, threshold):
    p = pd.read_csv(DATA / "organ_predictions.csv", dtype=str).fillna("")
    p = p[p.pred_organ != ""].copy()
    p["clip"] = p["pred_affected_part"].map(norm)
    eff = p["clip"]
    if (DATA / "class_part_labels.csv").exists():
        c = pd.read_csv(DATA / "class_part_labels.csv", dtype=str).fillna("")
        c = c[c["label"] != ""]
        eff = p["class"].map({r["class"]: norm(r["label"]) for _, r in c.iterrows()}).fillna(eff)
    if (DATA / "image_part_labels.csv").exists():
        i = pd.read_csv(DATA / "image_part_labels.csv", dtype=str).fillna("")
        i = i[i["parts"] != ""]
        eff = p["path"].map({r["path"]: ";".join(norm(x) for x in r["parts"].split(";"))
                             for _, r in i.iterrows()}).fillna(eff)
    p["eff"] = eff
    attrs = pd.read_csv(DATA / "class_attributes.csv", dtype=str).fillna("")
    healthy = set(attrs.loc[attrs["disease_type"] == "healthy", "class"])
    ok = (p["eff"] == part) & (p["clip"] == part) & (p["confidence"].astype(float) > threshold) \
        & ~p["class"].isin(healthy)
    return p[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="leaf")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--apply", action="store_true", help="write to the DB (default: dry run)")
    args = ap.parse_args()

    cand = candidates(args.part, args.threshold)
    print(f"{len(cand):,} images predicted '{args.part}' with confidence > {args.threshold}")

    url = os.environ.get("DB_URL") or sys.exit("Set DB_URL")
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    kwargs = {"pool_pre_ping": True}
    if url.startswith("postgresql+psycopg://"):  # psycopg3 + Supabase pooler
        kwargs["connect_args"] = {"prepare_threshold": None}
    engine = create_engine(url, **kwargs)

    with engine.connect() as c:
        done = {r[0] for r in c.execute(text("SELECT path FROM organ_review WHERE reviewed"))}
        todo = cand[~cand.path.isin(done)]
        print(f"{len(cand) - len(todo):,} already reviewed (left alone), {len(todo):,} to mark reviewed")
        print(todo.groupby("class").size().sort_values(ascending=False).head(10).to_string())
        if not args.apply:
            print("dry run - re-run with --apply to write")
            return
        sql = text("""
            INSERT INTO organ_review (path, class, pred_organ, confidence, final_label, reviewed, reviewed_at, label_source)
            VALUES (:path, :cls, :pred_organ, :confidence, :part, TRUE, :now, 'auto_conf')
            ON CONFLICT (path) DO UPDATE SET final_label = :part, reviewed = TRUE,
                reviewed_at = EXCLUDED.reviewed_at, label_source = 'auto_conf'
            WHERE organ_review.reviewed = FALSE
        """)
        now = datetime.now(timezone.utc)
        rows = [{"path": r.path, "cls": r["class"], "pred_organ": r.pred_organ,
                 "confidence": r.confidence, "part": args.part, "now": now}
                for _, r in todo.iterrows()]
        for i in range(0, len(rows), 2000):
            c.execute(sql, rows[i:i + 2000])
        c.commit()
        print(f"marked {len(rows):,} images reviewed as '{args.part}'")


if __name__ == "__main__":
    main()
