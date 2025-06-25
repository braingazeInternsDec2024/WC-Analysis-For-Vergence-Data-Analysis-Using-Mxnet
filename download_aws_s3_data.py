import boto3, botocore, os, pathlib, sys, textwrap

# Add dotenv import
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
    ENV_FILE_LOADED = True
except ImportError:
    ENV_FILE_LOADED = False

# Parameters
BUCKET      = "bgaze-odd-tasks-data"
PREFIX      = "odd-tasks-data/"          # keep trailing slash
LOCAL_DIR   = pathlib.Path("./data")
BUCKET_REGION = "eu-north-1"             # Stockholm
REQUESTER_PAYS = True                    # <- this bucket is RP

# Fetch AWS credentials
def fetch_keys_from_kaggle() -> tuple[str | None, str | None]:
    try:
        from kaggle_secrets import UserSecretsClient
        u = UserSecretsClient()
        return u.get_secret("AWS_ACCESS_KEY_ID"), u.get_secret("AWS_SECRET_ACCESS_KEY")
    except Exception:
        return (None, None)

AWS_KEY, AWS_SECRET = fetch_keys_from_kaggle()
AWS_KEY    = AWS_KEY    or os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = AWS_SECRET or os.getenv("AWS_SECRET_ACCESS_KEY")

# Build the S3 client
session = boto3.session.Session()
if AWS_KEY and AWS_SECRET:
    s3 = session.client("s3",
                        region_name=BUCKET_REGION,
                        aws_access_key_id=AWS_KEY,
                        aws_secret_access_key=AWS_SECRET)
else:
    s3 = session.client("s3", region_name=BUCKET_REGION)

# Extra header for Requester-Pays buckets
EXTRA = {"RequestPayer": "requester"} if REQUESTER_PAYS else None

def dl(bucket: str, key: str, dest: pathlib.Path):
    try:
        print(f"Downloading {key} to {dest} with extra args: {EXTRA}")
        s3.download_file(bucket, key, str(dest), ExtraArgs=EXTRA or {})
    except botocore.exceptions.ClientError as e:
        print(f"Error downloading {key}: {e}")
        if (e.response["Error"]["Code"] in ("403", "AccessDenied")) and not EXTRA:
            print(f"Retrying download of {key} with RequestPayer header")
            s3.download_file(bucket, key, str(dest),
                             ExtraArgs={"RequestPayer": "requester"})
        else:
            raise

def main() -> None:
    LOCAL_DIR.mkdir(exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    n = 0

    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):  # skip 'folder' placeholders
                continue

            rel  = pathlib.Path(key).relative_to(PREFIX)
            dest = LOCAL_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            print(f"📥  {key}  →  {dest}")
            dl(BUCKET, key, dest)
            n += 1

    print(f"\n✅  Finished – downloaded {n} object(s).")

if __name__ == "__main__":
    try:
        main()
    except botocore.exceptions.NoCredentialsError:
        env_file_msg = "\n• .env file: create a .env file with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY" if not ENV_FILE_LOADED else ""
        sys.exit(
            textwrap.dedent(f"""
            ❌  No AWS credentials found.

            • Kaggle:  add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
                       in the "Secrets" sidebar.{env_file_msg}
            • Local :  run `aws configure`  OR  export the two env-vars.
            """).strip()
        )