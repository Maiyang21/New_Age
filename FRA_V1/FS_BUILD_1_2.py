import os
import pandas as pd
import boto3 as bt3
import pyarrow.parquet as pq
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Initialize AWS Session and S3
session = bt3.Session()
fs_region = session.region_name
s3 = session.resource('s3')
bucket = 'foodsecbucket'

# Create bucket if it does not exist
try:
    if fs_region == 'us-east-1':
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={'LocationConstraint': fs_region})
    print(f'Bucket "{bucket}" successfully created')
except bt3.exceptions.BucketAlreadyOwnedByYou:
    print(f'Bucket "{bucket}" already exists.')
except Exception as e:
    print('Bucket creation failed:', e)

# Define S3 keys
output_prefix = 'FS_METRIC'
fs_out_key = f's3://{bucket}/{output_prefix}/out'
input_prefix = 'TRAIN_METRIC'
fs_train_key = f's3://{bucket}/{input_prefix}/training'
fs_test_key = f's3://{bucket}/{input_prefix}/testing'

# Load dataset
fs_train = pd.read_csv('C:/Users/PC/Documents/DATABASE/team datalink/FS_FeatureStore/FS_V1_prep.csv')

# Define time range
prediction_length = 10  # 10 years

general_start = fs_train['Year'].min()
general_end = fs_train['Year'].max()
context_length = general_end - general_start

# Ensure "start" column is a proper datetime format
fs_train['start'] = pd.to_datetime(fs_train['Year'].astype(str) + "-01-01")

dynamic_feature_columns = [
    'Average Caloric Value',
    'Average Person Calories',
    'Population',
    'Value_AVAIL',
    'AccessibilityMetric',
    'Value_P',
    'Value_PPP',
    'UtilizationMetric',
    'Value_SUA'
]

# Normalize dynamic features
scaler = MinMaxScaler()
fs_train[dynamic_feature_columns] = scaler.fit_transform(fs_train[dynamic_feature_columns])

# Replace NaN or infinite values with 0
fs_train[dynamic_feature_columns] = fs_train[dynamic_feature_columns].replace([np.inf, -np.inf], np.nan)
fs_train[dynamic_feature_columns] = fs_train[dynamic_feature_columns].fillna(0)

# Convert to 32-bit float to ensure compatibility
fs_train[dynamic_feature_columns] = fs_train[dynamic_feature_columns].astype(np.float32)

# Ensure target is a list of time-series values
fs_train['target'] = [fs_train['Availability Metric'].tolist()] * len(fs_train)

# Ensure dynamic features match target length
dynamic_feat_values = [fs_train[dynamic_feature_columns].T.values.tolist()] * len(fs_train)
fs_train['dynamic_feat'] = dynamic_feat_values

# Keep only necessary columns for DeepAR
fs_train = fs_train[['start', 'target', 'dynamic_feat']]

# Split dataset for training & testing
train_data = fs_train.iloc[: -prediction_length].copy()
test_data = fs_train.copy()

# Save as Parquet
train_parquet_path = "C:/Users/PC/Documents/DATABASE/team datalink/FS_model_formats/fs_train_v1.parquet"
test_parquet_path = "C:/Users/PC/Documents/DATABASE/team datalink/FS_model_formats/fs_test_v1.parquet"

train_data.to_parquet(train_parquet_path, engine='pyarrow', index=False)
test_data.to_parquet(test_parquet_path, engine='pyarrow', index=False)

print("Parquet files saved locally with 'start', 'target', and 'dynamic_features'.")


# Function to upload file to S3
def S3_upload(file_path, s3_path, override=True):
    split = s3_path.split('/')
    bucket_name = split[2]
    key = '/'.join(split[3:])
    bucket_obj = s3.Bucket(bucket_name)

    # Check if file already exists in S3
    objs = list(bucket_obj.objects.filter(Prefix=key))
    if objs:
        if not override:
            print(f"File s3://{bucket_name}/{key} already exists.\nSet override=True to upload anyway.\n")
            return
        else:
            print("Overwriting existing file")

    # Upload file to S3
    with open(file_path, "rb") as data:
        print(f"Uploading file to {s3_path}...")
        bucket_obj.put_object(Key=key, Body=data)


# Upload the Parquet files to S3
S3_upload(train_parquet_path, f"{fs_train_key}/train.parquet")
S3_upload(test_parquet_path, f"{fs_test_key}/test.parquet")

print("Data successfully converted to Parquet and uploaded for DeepAR!")
