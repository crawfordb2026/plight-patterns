#!/usr/bin/env python3
"""
AWS S3 sync utility for flyght-patterns.

Syncs monitor files, processed CSVs, and analysis results to/from an S3 bucket
so collaborators can share data across machines.

Setup (one-time):
  1. pip install boto3
  2. aws configure  (or set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY as vars in .env)
  3. Create Python/AWS/s3_config.json:
         {"bucket": "your-bucket-name"}

Usage:
  python s3_sync.py push --prefix experiment1     # upload all pipeline files from exp1 to S3 
  python s3_sync.py push --prefix experiment2     # upload all pipeline files from exp2 to S3
  python s3_sync.py pull --prefix experiment1     # download all pipeline files from S3-exp1

  python s3_sync.py status                        # compare local vs S3 for all categories

  Category flags (combine freely):
  python s3_sync.py push --raw        # only Monitors_raw/
  python s3_sync.py push --filtered   # only monitors_date_filtered/ + metadata.txt
  python s3_sync.py push --results    # only analyses/analysis_results/

  python s3_sync.py push --dry-run    # show what would be uploaded without doing it

S3 layout (relative to bucket root, or prefix/ if configured):
  monitors_raw/Monitor51.txt
  monitors_date_filtered/Monitor51_06_01_25.txt
  metadata.txt
  analysis_results/windowed_features.csv
  analysis_results/model_performance.txt
  analysis_results/feature_importances.csv
  analysis_results/predicted_vs_actual.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

try:
    from dotenv import load_dotenv
    # Look for .env in the repo root (two levels up from Python/AWS/)
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if not _env_path.exists():
        _env_path = Path(__file__).parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed; fall back to system env vars / ~/.aws/credentials

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# PYTHON_DIR = .../Python/  (parent of AWS/, where monitors and src/ live)
PYTHON_DIR = Path(__file__).parent.parent.resolve()

# Category definitions: local directory and corresponding S3 prefix.
# Files matching `exts` are synced; `recursive` controls subdirectory walking.
CATEGORIES = {
    "raw": {
        "local_dir": PYTHON_DIR / "Monitors_raw",
        "s3_prefix": "monitors_raw",
        "exts": {".txt"},
        "recursive": False,
    },
    "filtered": {
        "local_dir": PYTHON_DIR / "monitors_date_filtered",
        "s3_prefix": "monitors_date_filtered",
        "exts": {".txt"},
        "recursive": False,
    },
    "results": {
        "local_dir": PYTHON_DIR.parent / "analyses" / "analysis_results",
        "s3_prefix": "analysis_results",
        "exts": {".csv", ".png", ".pdf", ".txt"},
        "recursive": True,
    },
}

# metadata.txt lives at Python/metadata.txt, not inside a category directory.
METADATA_LOCAL = PYTHON_DIR / "metadata.txt"
METADATA_S3_RELKEY = "metadata.txt"

CONFIG_PATH = Path(__file__).parent / "s3_config.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(bucket_override=None, prefix_override=None):
    if not CONFIG_PATH.exists():
        if bucket_override:
            return {"bucket": bucket_override, "prefix": prefix_override or ""}
        print(f"ERROR: Config file not found: {CONFIG_PATH}")
        print("Create it with:")
        print('  {"bucket": "your-bucket-name"}')
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    if "bucket" not in cfg:
        print("ERROR: s3_config.json must contain a 'bucket' key.")
        sys.exit(1)
    if bucket_override:
        cfg["bucket"] = bucket_override
    if prefix_override is not None:
        cfg["prefix"] = prefix_override
    cfg.setdefault("prefix", "")
    return cfg


# ---------------------------------------------------------------------------
# S3 key helpers
# ---------------------------------------------------------------------------

def make_s3_key(prefix, *parts):
    """Join prefix and path parts into an S3 key, ignoring empty segments."""
    segments = [p for p in [prefix] + list(parts) if p]
    return "/".join(segments)


def list_local_for_push(cat, global_prefix):
    """Yield (local_path, s3_key) pairs for files that exist locally."""
    local_dir = cat["local_dir"]
    if not local_dir.exists():
        return
    walker = local_dir.rglob("*") if cat["recursive"] else local_dir.glob("*")
    for f in walker:
        if f.is_file() and f.suffix in cat["exts"]:
            rel = f.relative_to(local_dir).as_posix()
            yield f, make_s3_key(global_prefix, cat["s3_prefix"], rel)


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

def push_entry(s3, bucket, local_path, s3_key, dry_run):
    """Upload one file if it is absent from S3 or differs in size. Returns True if uploaded."""
    local_size = local_path.stat().st_size
    try:
        obj = s3.head_object(Bucket=bucket, Key=s3_key)
        if obj["ContentLength"] == local_size:
            return False  # already in sync
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            raise

    print(f"  PUSH  {local_path.name}  ->  s3://{bucket}/{s3_key}")
    if not dry_run:
        s3.upload_file(str(local_path), bucket, s3_key)
    return True


def push_category(s3, bucket, global_prefix, cat_name, cat, dry_run):
    entries = list(list_local_for_push(cat, global_prefix))
    if not entries:
        print(f"  [{cat_name}] no local files found — skipped")
        return 0, 0
    uploaded = skipped = 0
    for local_path, s3_key in entries:
        if push_entry(s3, bucket, local_path, s3_key, dry_run):
            uploaded += 1
        else:
            skipped += 1
    return uploaded, skipped


def push_metadata(s3, bucket, global_prefix, dry_run):
    if not METADATA_LOCAL.exists():
        print("  [metadata] metadata.txt not found — skipped")
        return 0, 0
    s3_key = make_s3_key(global_prefix, METADATA_S3_RELKEY)
    if push_entry(s3, bucket, METADATA_LOCAL, s3_key, dry_run):
        return 1, 0
    return 0, 1


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

def pull_category(s3, bucket, global_prefix, cat_name, cat, dry_run):
    s3_prefix = make_s3_key(global_prefix, cat["s3_prefix"])
    prefix_with_slash = s3_prefix + "/" if s3_prefix else ""
    local_dir = cat["local_dir"]

    downloaded = skipped = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_with_slash):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            remote_size = obj["Size"]
            # Reconstruct local path from the S3 key
            rel = key[len(prefix_with_slash):]
            if not rel:  # skip the prefix directory itself
                continue
            local_path = local_dir / Path(rel)
            if local_path.exists() and local_path.stat().st_size == remote_size:
                skipped += 1
                continue
            print(f"  PULL  s3://{bucket}/{key}  ->  {local_path.name}")
            if not dry_run:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(local_path))
            downloaded += 1
    if downloaded == 0 and skipped == 0:
        print(f"  [{cat_name}] nothing on S3 yet")
    return downloaded, skipped


def pull_metadata(s3, bucket, global_prefix, dry_run):
    s3_key = make_s3_key(global_prefix, METADATA_S3_RELKEY)
    try:
        obj = s3.head_object(Bucket=bucket, Key=s3_key)
        remote_size = obj["ContentLength"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            print("  [metadata] not on S3 yet")
            return 0, 0
        raise
    if METADATA_LOCAL.exists() and METADATA_LOCAL.stat().st_size == remote_size:
        return 0, 1
    print(f"  PULL  s3://{bucket}/{s3_key}  ->  metadata.txt")
    if not dry_run:
        s3.download_file(bucket, s3_key, str(METADATA_LOCAL))
    return 1, 0


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def status_category(s3, bucket, global_prefix, cat_name, cat):
    print(f"\n[{cat_name}]  local: {cat['local_dir']}")
    s3_prefix = make_s3_key(global_prefix, cat["s3_prefix"])
    prefix_with_slash = s3_prefix + "/" if s3_prefix else ""

    # Collect remote objects
    remote = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_with_slash):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix_with_slash):]
            if rel:
                remote[rel] = obj["Size"]

    # Collect local files
    local = {}
    if cat["local_dir"].exists():
        walker = cat["local_dir"].rglob("*") if cat["recursive"] else cat["local_dir"].glob("*")
        for f in walker:
            if f.is_file() and f.suffix in cat["exts"]:
                rel = f.relative_to(cat["local_dir"]).as_posix()
                local[rel] = f.stat().st_size

    all_keys = sorted(set(local) | set(remote))
    if not all_keys:
        print("  (empty)")
        return

    for rel in all_keys:
        local_size = local.get(rel)
        remote_size = remote.get(rel)
        if local_size is not None and remote_size is not None:
            tag = "OK  " if local_size == remote_size else f"DIFF (local={local_size}B s3={remote_size}B)"
        elif local_size is not None:
            tag = "LOCAL ONLY"
        else:
            tag = "S3 ONLY"
        print(f"  {tag:40s}  {rel}")


def status_metadata(s3, bucket, global_prefix):
    print("\n[metadata]")
    s3_key = make_s3_key(global_prefix, METADATA_S3_RELKEY)
    local_exists = METADATA_LOCAL.exists()
    local_size = METADATA_LOCAL.stat().st_size if local_exists else None
    try:
        obj = s3.head_object(Bucket=bucket, Key=s3_key)
        remote_size = obj["ContentLength"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            remote_size = None
        else:
            raise
    if local_size is not None and remote_size is not None:
        tag = "OK  " if local_size == remote_size else f"DIFF (local={local_size}B s3={remote_size}B)"
    elif local_size is not None:
        tag = "LOCAL ONLY"
    elif remote_size is not None:
        tag = "S3 ONLY"
    else:
        tag = "NOT FOUND"
    print(f"  {tag:40s}  metadata.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync flyght-patterns pipeline files to/from S3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command", choices=["push", "pull", "status"],
        help="push = upload to S3 | pull = download from S3 | status = compare",
    )
    parser.add_argument("--bucket", help="S3 bucket (overrides s3_config.json)")
    parser.add_argument("--prefix", default=None,
                        help="S3 key prefix, e.g. 'experiment-june-2025' (overrides config)")
    parser.add_argument("--raw", action="store_true", help="Include Monitors_raw/")
    parser.add_argument("--filtered", action="store_true",
                        help="Include monitors_date_filtered/ and metadata.txt")
    parser.add_argument("--results", action="store_true",
                        help="Include analyses/analysis_results/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without uploading or downloading")
    return parser.parse_args()


def resolve_active(args):
    """Return which categories and whether metadata should be included."""
    any_flag = args.raw or args.filtered or args.results
    active_cats = {
        "raw": args.raw or not any_flag,
        "filtered": args.filtered or not any_flag,
        "results": args.results or not any_flag,
    }
    include_metadata = args.filtered or not any_flag
    return active_cats, include_metadata


def connect_s3(bucket):
    try:
        s3 = boto3.client("s3")
        s3.head_bucket(Bucket=bucket)
        return s3
    except NoCredentialsError:
        print("ERROR: AWS credentials not found.")
        print("Run `aws configure` or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        sys.exit(1)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("403", "404", "NoSuchBucket"):
            print(f"ERROR: Cannot access bucket '{bucket}': {e}")
            sys.exit(1)
        raise


def main():
    args = parse_args()
    cfg = load_config(bucket_override=args.bucket, prefix_override=args.prefix)
    bucket = cfg["bucket"]
    prefix = cfg["prefix"]
    active_cats, include_metadata = resolve_active(args)

    s3 = connect_s3(bucket)

    print(f"Bucket: s3://{bucket}" + (f"/{prefix}" if prefix else ""))
    if args.dry_run:
        print("DRY RUN — no files will be transferred")

    total_changed = total_skipped = 0

    if args.command == "push":
        if include_metadata:
            c, s = push_metadata(s3, bucket, prefix, args.dry_run)
            total_changed += c; total_skipped += s
        for cat_name, enabled in active_cats.items():
            if not enabled:
                continue
            print(f"\n[{cat_name}]")
            c, s = push_category(s3, bucket, prefix, cat_name, CATEGORIES[cat_name], args.dry_run)
            total_changed += c; total_skipped += s
        action = "uploaded"

    elif args.command == "pull":
        if include_metadata:
            c, s = pull_metadata(s3, bucket, prefix, args.dry_run)
            total_changed += c; total_skipped += s
        for cat_name, enabled in active_cats.items():
            if not enabled:
                continue
            print(f"\n[{cat_name}]")
            c, s = pull_category(s3, bucket, prefix, cat_name, CATEGORIES[cat_name], args.dry_run)
            total_changed += c; total_skipped += s
        action = "downloaded"

    else:  # status
        status_metadata(s3, bucket, prefix)
        for cat_name, enabled in active_cats.items():
            if not enabled:
                continue
            status_category(s3, bucket, prefix, cat_name, CATEGORIES[cat_name])
        return

    print(f"\nDone: {total_changed} {action}, {total_skipped} already in sync.")


if __name__ == "__main__":
    main()
