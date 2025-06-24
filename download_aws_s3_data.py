import boto3
import os

# ✅ Get secrets using Kaggle's method
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    AWS_ACCESS_KEY = user_secrets.get_secret("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = user_secrets.get_secret("AWS_SECRET_ACCESS_KEY")
except:
    # Fallback for local dev using .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
        AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    except ImportError:
        AWS_ACCESS_KEY = AWS_SECRET_KEY = None

# Validate credentials
if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise EnvironmentError("AWS credentials not found.")

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

bucket_name = 'bgaze-odd-tasks-data'
prefix = 'odd-tasks-data/'

local_dir = './data'
os.makedirs(local_dir, exist_ok=True)

# List and download all files under the prefix
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
    if 'Contents' in page:
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('/'):
                continue
            relative_path = os.path.relpath(key, prefix)
            local_file_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            print(f"Downloading {key} to {local_file_path}")
            s3.download_file(bucket_name, key, local_file_path)

print("Download complete.")
