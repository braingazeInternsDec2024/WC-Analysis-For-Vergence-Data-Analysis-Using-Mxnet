# ===================== 0. Imports & Secrets  =====================
import boto3, os, pathlib, sys
from kaggle_secrets import UserSecretsClient

# Fetch AWS keys from the Kaggle “Secrets” sidebar
u = UserSecretsClient()
os.environ["AWS_ACCESS_KEY_ID"]     = u.get_secret("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = u.get_secret("AWS_SECRET_ACCESS_KEY")

AWS_KEY    = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]

# ===================== 1. Bucket parameters  =====================
BUCKET = "bgaze-odd-tasks-data"
PREFIX = "odd-tasks-data/"          # keep the trailing slash
LOCAL  = pathlib.Path("./data")     # where to save the files
REGION = "eu-north-1"               # ← bucket lives in Stockholm

# ===================== 2. Build the S3 clients ===================
# “Light” client in the correct region
s3 = boto3.client(
        "s3",
        region_name      = REGION,
        aws_access_key_id     = AWS_KEY,
        aws_secret_access_key = AWS_SECRET
)

# Optional: check if bucket is Requester-Pays
EXTRA_ARGS = None
try:
    payer_flag = s3.get_bucket_request_payment(Bucket=BUCKET)["Payer"]
    if payer_flag == "Requester":
        EXTRA_ARGS = {"RequestPayer": "requester"}
        print("Bucket is Requester-Pays – will send RequestPayer=requester.")
except s3.exceptions.ClientError as e:
    # Lack of permission ➜ just continue without the header
    print("Could not query RequestPayment (likely not allowed) – continuing without it.")

# Helper wrapper so we can call the same way in the loop
def download(bucket, key, dest):
    if EXTRA_ARGS:
        s3.download_file(bucket, key, str(dest), ExtraArgs=EXTRA_ARGS)
    else:
        s3.download_file(bucket, key, str(dest))

# ===================== 3. Download loop ==========================
LOCAL.mkdir(exist_ok=True)
paginator = s3.get_paginator("list_objects_v2")

n_files = 0
for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
    for obj in page.get("Contents", []):
        key = obj["Key"]

        # Skip pseudo-folders
        if key.endswith("/"):
            continue

        # Build local path
        rel  = pathlib.Path(key).relative_to(PREFIX)
        dest = LOCAL / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        print(f"📥  {key}  →  {dest}")
        download(BUCKET, key, dest)
        n_files += 1

print(f"\n✅  All done – downloaded {n_files} objects.")