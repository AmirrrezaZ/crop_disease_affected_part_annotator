"""Verify that every image of plantwild (v1) + plantwild_v2 is in the Supabase bucket.

Compares three sets of object keys (= relative paths like
"plantwild/images/apple scab/1.jpg"):
  local   every image file under <source>/plantwild/images and <source>/plantwild_v2
  preds   every path in data/organ_predictions.csv (what the app asks the bucket for)
  bucket  every object actually listed in the bucket (S3 ListObjectsV2)
and prints what is missing from the bucket. Optionally (--http-sample N) also
GETs N random public URLs to prove they are really readable by the app.

Missing keys are written to missing_from_bucket.txt; re-run upload_images.py
(it skips existing keys) to upload only those.

Run:
  .venv/bin/python scripts/check_bucket.py [--http-sample 30]
"""
import argparse
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "crop_disease" / "version1" / "data"
IMAGE_DIRS = ["plantwild/images", "plantwild_v2"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".jfif"}  # same as upload_images.py


def local_keys(source: Path) -> dict[str, set]:
    out = {}
    for d in IMAGE_DIRS:
        base = source / d
        if not base.exists():
            sys.exit(f"missing local dir {base}")
        out[d] = {str(f.relative_to(source)) for f in base.rglob("*") if f.suffix.lower() in EXTS}
    return out


def bucket_keys() -> set:
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["SUPABASE_S3_ENDPOINT"],
        aws_access_key_id=os.environ["SUPABASE_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["SUPABASE_S3_SECRET_ACCESS_KEY"],
        region_name=os.environ["SUPABASE_S3_REGION"],
    )
    keys = set()
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=os.environ["SUPABASE_BUCKET"])
    for page in pages:
        keys.update(o["Key"] for o in page.get("Contents", []))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--http-sample", type=int, default=0,
                    help="also fetch N random public URLs (uses IMAGE_PUBLIC_BASE)")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    local_by_dir = local_keys(Path(args.source))
    local = set().union(*local_by_dir.values())
    preds = set(pd.read_csv(ROOT / "data" / "organ_predictions.csv", dtype=str)["path"])
    bucket = bucket_keys()

    print(f"{'set':34s}{'count':>8s}")
    for d, ks in local_by_dir.items():
        print(f"local {d:28s}{len(ks):8d}   in bucket: {len(ks & bucket)}")
    print(f"{'local total':34s}{len(local):8d}   in bucket: {len(local & bucket)}")
    print(f"{'organ_predictions.csv (app)':34s}{len(preds):8d}   in bucket: {len(preds & bucket)}")
    print(f"{'bucket objects':34s}{len(bucket):8d}")

    missing_local = sorted(local - bucket)
    missing_app = sorted(preds - bucket)
    extra = sorted(bucket - local - preds)
    print(f"\nlocal images missing from bucket : {len(missing_local)}")
    print(f"app (CSV) paths missing in bucket : {len(missing_app)}")
    print(f"bucket objects not in local/CSV   : {len(extra)}")
    print(f"CSV paths with no local file      : {len(preds - local)}")
    print(f"local images not in the CSV       : {len(local - preds)}")

    if missing_local or missing_app:
        by_class = Counter(Path(k).parent.name for k in set(missing_local) | set(missing_app))
        print("\nmissing by class:", dict(by_class.most_common(15)))
        out = ROOT / "missing_from_bucket.txt"
        out.write_text("\n".join(sorted(set(missing_local) | set(missing_app))) + "\n")
        print(f"full list -> {out.name}")

    if args.http_sample:
        base = os.environ["IMAGE_PUBLIC_BASE"].rstrip("/")
        sample = random.sample(sorted(preds), min(args.http_sample, len(preds)))

        def head(k):
            try:
                return requests.get(f"{base}/{k}", timeout=30, stream=True).status_code
            except Exception as e:
                return type(e).__name__

        with ThreadPoolExecutor(8) as ex:
            codes = Counter(ex.map(head, sample))
        print(f"\nHTTP sample of {len(sample)} public URLs:", dict(codes))

    ok = not missing_local and not missing_app
    print("\nRESULT:", "ALL images are in the bucket" if ok else "SOME images are MISSING")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
