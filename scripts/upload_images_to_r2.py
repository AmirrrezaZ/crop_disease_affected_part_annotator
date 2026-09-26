"""One-time upload of crop_disease/version1's image dataset (plantwild +
plantwild_v2, ~4.2GB / 22.7k files) to a Cloudflare R2 bucket, preserving
the same relative path used in data/organ_predictions.csv as the object
key, so the deployed app's image_url() maps 1:1 onto R2's public URL.

Resumable: skips any key that already exists in the bucket, so re-running
after a network drop picks up where it left off instead of re-uploading
everything.

Requires (R2 dashboard -> Manage API tokens -> create S3 API token):
  R2_ACCOUNT_ID
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_BUCKET

Run:
  pip install boto3
  R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=... \\
    python scripts/upload_images_to_r2.py [--source /path/to/crop_disease/version1/data]
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

IMAGE_DIRS = ["plantwild/images", "plantwild_v2"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".jfif"}


def make_client():
    required = ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        sys.exit(f"missing env vars: {missing}")
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def key_exists(client, bucket, key):
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def upload_one(client, bucket, local_path, key):
    if key_exists(client, bucket, key):
        return "skipped"
    content_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    }.get(local_path.suffix.lower(), "application/octet-stream")
    client.upload_file(str(local_path), bucket, key, ExtraArgs={"ContentType": content_type})
    return "uploaded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(
        Path(__file__).resolve().parents[2] / "crop_disease" / "version1" / "data"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    source = Path(args.source)
    bucket = os.environ["R2_BUCKET"]
    client = make_client()

    files = []
    for image_dir in IMAGE_DIRS:
        base = source / image_dir
        if not base.exists():
            print(f"skip missing dir: {base}")
            continue
        for f in base.rglob("*"):
            if f.suffix.lower() in EXTS:
                key = str(f.relative_to(source))
                files.append((f, key))
    print(f"found {len(files)} images under {source}")

    uploaded = skipped = failed = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(upload_one, client, bucket, f, k): k for f, k in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="uploading"):
            key = futures[fut]
            try:
                result = fut.result()
                uploaded += result == "uploaded"
                skipped += result == "skipped"
            except Exception as e:
                failed += 1
                print(f"FAILED {key}: {e}")

    print(f"\nuploaded={uploaded} skipped(already present)={skipped} failed={failed}")
    if failed:
        print("re-run the same command to retry only the failed/missing ones.")


if __name__ == "__main__":
    main()
