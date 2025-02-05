# import firebase_admin
# from firebase_admin import credentials, storage
# import os

# # Initialize Firebase Admin SDK
# def initialize_firebase_app(service_account_key_path):
#     if not firebase_admin._apps:
#         cred = credentials.Certificate(service_account_key_path)
#         firebase_admin.initialize_app(cred, {
#             'storageBucket': 'oddballtask-3a436.firebasestorage.app'
#         })


# # Recursively list all files in a Firebase Storage bucket
# def list_files(bucket, prefix=""):
#     blobs = bucket.list_blobs(prefix=prefix)
#     files = []
#     for blob in blobs:
#         files.append(blob.name)
#     return files

# # Download a file from Firebase Storage
# def download_file(blob, destination_path):
#     os.makedirs(os.path.dirname(destination_path), exist_ok=True)
#     blob.download_to_filename(destination_path)
#     print(f"Downloaded: {blob.name} -> {destination_path}")

# # Download all files and folders from Firebase Storage
# def download_all_files(bucket_name, destination_folder):
#     bucket = storage.bucket(bucket_name)
#     files = list_files(bucket)

#     for file_path in files:
#         blob = bucket.blob(file_path)
#         local_file_path = os.path.join(destination_folder, file_path)
#         download_file(blob, local_file_path)

# # Main function
# if __name__ == "__main__":
#     # Path to your service account key file
#     service_account_key_path = "oddballtask-3a436-firebase-adminsdk-fbsvc-ae72113f50.json"

#     # Initialize Firebase
#     initialize_firebase_app(service_account_key_path)

#     # Define the bucket name and local destination folder
#     bucket_name = "oddballtask-3a436.firebasestorage.app"

#     destination_folder = "data"  

#     # Download all files and folders
#     download_all_files(bucket_name, destination_folder)