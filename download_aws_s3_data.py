import boto3, os
from kaggle_secrets import UserSecretsClient

# 1️⃣ credentials from Kaggle secrets
u = UserSecretsClient()
os.environ['AWS_ACCESS_KEY_ID']     = u.get_secret("AWS_ACCESS_KEY_ID")
os.environ['AWS_SECRET_ACCESS_KEY'] = u.get_secret("AWS_SECRET_ACCESS_KEY")

bucket  = 'bgaze-odd-tasks-data'
prefix  = 'odd-tasks-data/'
local   = './data'

# 2️⃣ detect region & requester-pays
s3_tmp   = boto3.client('s3',
                        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
region   = s3_tmp.get_bucket_location(Bucket=bucket)['LocationConstraint']
payer    = s3_tmp.get_bucket_request_payment(Bucket=bucket)['Payer'] == 'Requester'
extra    = {'RequestPayer': 'requester'} if payer else None

# 3️⃣ final client in correct region
s3 = boto3.client('s3',
                  region_name=region,
                  aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                  aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])

os.makedirs(local, exist_ok=True)
paginator = s3.get_paginator('list_objects_v2')

for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get('Contents', []):
        key = obj['Key']
        if key.endswith('/'):
            continue
        rel  = os.path.relpath(key, prefix)
        dest = os.path.join(local, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        print(f"Downloading {key} → {dest}")
        if extra:
            s3.download_file(bucket, key, dest, ExtraArgs=extra)
        else:
            s3.download_file(bucket, key, dest)

print("✅ All done!")