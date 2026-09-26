"""One-time upload of crop_disease/version1's image dataset (plantwild +
plantwild_v2, ~22.7k files, 4.2GB at full res) to Supabase Storage, via its
S3-compatible API. Images are downscaled on the way up -- this app is for
organ identification (leaf/stem/fruit/flower/root/...), not symptom-level
diagnosis, so a small thumbnail is plenty, and it keeps the whole dataset
comfortably inside Supabase's free-tier 1GB storage cap (~0.48GB at the
defaults below vs. 1GB limit).

Preserves the same relative path used in data/organ_predictions.csv as the
object key, so the deployed app's image_url() maps 1:1 onto the bucket's
public URL. Resumable: skips any key that already exists, so re-running
after a network drop picks up where it left off.

Requires (Supabase dashboard -> Project Settings -> Storage -> S3 Connection
-> "New access key"):
  SUPABASE_S3_ENDPOINT     e.g. https://<project-ref>.supabase.co/storage/v1/s3
  SUPABASE_S3_REGION       shown on the same page, e.g. us-east-1
  SUPABASE_S3_ACCESS_KEY_ID
  SUPABASE_S3_SECRET_ACCESS_KEY
  SUPABASE_BUCKET          the bucket name (create it first, mark it Public)

Run:
  pip install -r scripts/requirements-migration.txt
  SUPABASE_S3_ENDPOINT=... SUPABASE_S3_REGION=... SUPABASE_S3_ACCESS_KEY_ID=... \\
    SUPABASE_S3_SECRET_ACCESS_KEY=... SUPABASE_BUCKET=crop-disease-images \\
    python scripts/upload_images.py [--source /path/to/crop_disease/version1/data] \\
    [--max-dim 384] [--quality 78]
"""
import argparse
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from PIL import Image, ImageOps
from tqdm import tqdm

IMAGE_DIRS = ["plantwild/images", "plantwild_v2"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".jfif"}


def make_client():
    required = ["SUPABASE_S3_ENDPOINT", "SUPABASE_S3_REGION", "SUPABASE_S3_ACCESS_KEY_ID",
                "SUPABASE_S3_SECRET_ACCESS_KEY", "SUPABASE_BUCKET"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        sys.exit(f"missing env vars: {missing}")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SUPABASE_S3_ENDPOINT"],
        aws_access_key_id=os.environ["SUPABASE_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["SUPABASE_S3_SECRET_ACCESS_KEY"],
        region_name=os.environ["SUPABASE_S3_REGION"],
    )


def key_exists(client, bucket, key):
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def downscale(local_path: Path, max_dim: int, quality: int) -> bytes:
    im = Image.open(local_path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def upload_one(client, bucket, local_path, key, max_dim, quality):
    if key_exists(client, bucket, key):
        return "skipped"
    data = downscale(local_path, max_dim, quality)
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType="image/jpeg")
    return "uploaded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(
        Path(__file__).resolve().parents[2] / "crop_disease" / "version1" / "data"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-dim", type=int, default=384,
                     help="longest-side pixels after resize (384 -> ~0.48GB total; "
                          "512 -> ~0.81GB, closer to the 1GB free-tier cap)")
    ap.add_argument("--quality", type=int, default=78)
    args = ap.parse_args()

    source = Path(args.source)
    bucket = os.environ["SUPABASE_BUCKET"]
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
    print(f"found {len(files)} images under {source}; resizing to max {args.max_dim}px, q{args.quality}")

    uploaded = skipped = failed = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(upload_one, client, bucket, f, k, args.max_dim, args.quality): k
                   for f, k in files}
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
