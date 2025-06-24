#!/usr/bin/env python3
"""
Download every object under odd-tasks-data/ from the bucket
bgaze-odd-tasks-data and store them under ./data/…

• Works on Kaggle:   put your keys in the “Secrets” sidebar
• Works locally:     `aws configure`  OR  `export AWS_ACCESS_KEY_ID=…`

"""
from __future__ import annotations
import boto3, botocore, os, pathlib, sys, textwrap

# ---------------------------------------------------------------------
# 0. Parameters — edit these if you fork this script
# ---------------------------------------------------------------------
BUCKET      = "bgaze-odd-tasks-data"
PREFIX      = "odd-tasks-data/"          # keep trailing slash
LOCAL_DIR   = pathlib.Path("./data")
BUCKET_REGION = "eu-north-1"             # Stockholm
REQUESTER_PAYS = True                    # <- this bucket is RP

# ---------------------------------------------------------------------
# 1. Get AWS credentials (Kaggle Secrets  ➜  env vars  ➜  ~/.aws/…)
# ---------------------------------------------------------------------
def fetch_keys_from_kaggle() -> tuple[str | None, str | None]:
    """
    Safely try Kaggle's UserSecretsClient. Return (None, None)
    if we are NOT on Kaggle or no secrets exist.
    """
    try:
        from kaggle_secrets import UserSecretsClient
        u = UserSecretsClient()
        return u.get_secret("AWS_ACCESS_KEY_ID"), u.get_secret("AWS_SECRET_ACCESS_KEY")
    except Exception:
        return (None, None)

AWS_KEY, AWS_SECRET = fetch_keys_from_kaggle()
AWS_KEY    = AWS_KEY    or os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = AWS_SECRET or os.getenv("AWS_SECRET_ACCESS_KEY")

# ---------------------------------------------------------------------
# 2. Build the S3 client in the correct region
# ---------------------------------------------------------------------
session = boto3.session.Session()
if AWS_KEY and AWS_SECRET:
    s3 = session.client("s3",
                        region_name=BUCKET_REGION,
                        aws_access_key_id=AWS_KEY,
                        aws_secret_access_key=AWS_SECRET)
else:
    # Fall back to profile / IAM role
    s3 = session.client("s3", region_name=BUCKET_REGION)

# Extra header for Requester-Pays buckets
EXTRA = {"RequestPayer": "requester"} if REQUESTER_PAYS else None

def dl(bucket: str, key: str, dest: pathlib.Path):
    """
    One wrapper around download_file so every request carries
    RequestPayer=requester. Retries once if we forgot the header.
    """
    try:
        s3.download_file(bucket, key, str(dest), ExtraArgs=EXTRA or {})
    except botocore.exceptions.ClientError as e:
        if (e.response["Error"]["Code"] in ("403", "AccessDenied")) and not EXTRA:
            # Retry with RequestPayer — should never happen in this script
            s3.download_file(bucket, key, str(dest),
                             ExtraArgs={"RequestPayer": "requester"})
        else:
            raise

# ---------------------------------------------------------------------
# 3. Download loop
# ---------------------------------------------------------------------
def main() -> None:
    LOCAL_DIR.mkdir(exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    n = 0

    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):              # skip 'folder' placeholders
                continue

            rel  = pathlib.Path(key).relative_to(PREFIX)
            dest = LOCAL_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            print(f"📥  {key}  →  {dest}")
            dl(BUCKET, key, dest)
            n += 1

    print(f"\n✅  Finished – downloaded {n} object(s).")

# ---------------------------------------------------------------------
# 4. CLI entry-point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except botocore.exceptions.NoCredentialsError:
        sys.exit(
            textwrap.dedent("""
            ❌  No AWS credentials found.

            • Kaggle:  add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
                       in the “Secrets” sidebar.
            • Local :  run `aws configure`  OR  export the two env-vars.
            """).strip()
        )