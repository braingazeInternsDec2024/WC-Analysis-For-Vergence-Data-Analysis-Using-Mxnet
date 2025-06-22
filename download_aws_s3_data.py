import boto3
import os

# Initialize S3 client
s3 = boto3.client('s3')

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
