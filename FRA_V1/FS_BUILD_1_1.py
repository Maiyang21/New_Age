import os
import json
import pandas as pd
import boto3 as bt3
import sagemaker as sage

"""MODEL BUILD WITH AWS CLOUD"""

# Environment setup, S3 bucket prep, and session creation
fs_region = bt3.session.Session().region_name
s3 = bt3.resource('s3')
bucket = 'foodsecbucket1'

try:
    if fs_region == 'us-east-1':
        s3.create_bucket(Bucket=bucket)
        print('Bucket successfully created')
except Exception as e:
    print('Bucket creation failed:', e)

# Bucket object key creation
output_prefix = 'FS_METRIC'
fs_out_key = f's3://{bucket}/{output_prefix}/out'
input_prefix = 'TRAIN_METRIC'
fs_train_key = f's3://{bucket}/{input_prefix}/training'
fs_test_key = f's3://{bucket}/{input_prefix}/testing'

# Data Formatting
fs_train = pd.read_csv('C:/Users/PC/Documents/DATABASE/team datalink/FS_FeatureStore/FS_V1_prep.csv')

freq = 'Y'
prediction_length = 10  # 10 years
general_start = fs_train['Year'].min()
general_end = fs_train['Year'].max()
context_length = general_end - general_start

# Proper slicing for target values
train_target = fs_train['Availability Metric'].iloc[: -prediction_length].tolist()
test_target = fs_train['Availability Metric'].tolist()

Train_data = [{
    "start": str(general_start),
    "target": train_target,
    "dynamic_feat": [
        fs_train['Average Caloric Value'].tolist(),
        fs_train['Average Person Calories'].tolist(),
        fs_train['Population'].tolist(),
        fs_train['Value_AVAIL'].tolist(),
        fs_train['AccessibilityMetric'].tolist(),
        fs_train['Value_P'].tolist(),
        fs_train['Value_PPP'].tolist(),
        fs_train['UtilizationMetric'].tolist(),
        fs_train['Value_SUA'].tolist()
    ]
}]

Test_data = [{
    "start": str(general_start),
    "target": test_target,
    "dynamic_feat": [
        fs_train['Average Caloric Value'].tolist(),
        fs_train['Average Person Calories'].tolist(),
        fs_train['Population'].tolist(),
        fs_train['Value_AVAIL'].tolist(),
        fs_train['AccessibilityMetric'].tolist(),
        fs_train['Value_P'].tolist(),
        fs_train['Value_PPP'].tolist(),
        fs_train['UtilizationMetric'].tolist(),
        fs_train['Value_SUA'].tolist()
    ]
}]


# Function to convert JSON dict format to file (NDJSON)
def write_dicts_to_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            json.dump(d, f)  # Correctly serialize JSON
            f.write("\n")  # Ensure NDJSON format


# Implementing JSON conversion function
train_json_path = "C:/Users/PC/Documents/DATABASE/team datalink/FS_model_formats/fs_train_v1.json"
test_json_path = "C:/Users/PC/Documents/DATABASE/team datalink/FS_model_formats/fs_test_v1.json"

write_dicts_to_file(train_json_path, Train_data)
write_dicts_to_file(test_json_path, Test_data)


# Function to upload preprocessed data to S3
def S3_upload(file_path, s3_path, override=False):
    split = s3_path.split('/')
    bucket_name = split[2]
    key = '/'.join(split[3:])
    bucket_obj = s3.Bucket(bucket_name)

    # Check if file already exists in S3
    if any(bucket_obj.objects.filter(Prefix=key)):
        if not override:
            print(f"File s3://{bucket_name}/{key} already exists.\nSet override=True to upload anyway.\n")
            return
        else:
            print("Overwriting existing file")

    # Upload file to S3
    with open(file_path, "rb") as data:
        print(f"Uploading file to {s3_path}...")
        bucket_obj.put_object(Key=key, Body=data)


# Implementing JSON cloud upload
S3_upload(train_json_path, f"{fs_train_key}/train.json")
S3_upload(test_json_path, f"{fs_test_key}/test.json")

print("Data successfully processed and uploaded!")

# Model initialization
image_name = image_uris.retrieve(framework="forecasting-deepar", region=fs_region)  # container call on global
sagemaker_session = sage.Session()
role = "arn:aws:iam::021891608166:role/FS-exec"  # IAM role for sagemaker

estimator = sage.estimator.Estimator(
    image_uri=image_name,
    sagemaker_session=sagemaker_session,
    role=role,
    instance_count=1,
    instance_type="ml.c4.2xlarge",
    base_job_name="deepar-FoodSecurity-v1",
    output_path=fs_out_key,
    use_spot_instances=True,
    max_wait=7200,
    max_run=3600
)


# setting up Model Tunes
hyperparameters = {
    "time_freq": freq,
    "epochs": "40",
    "early_stopping_patience": "40",
    "learning_rate": "5E-4",
    "context_length": str(context_length),
    "prediction_length": str(prediction_length),
}
estimator.set_hyperparameters(**hyperparameters)

# Model training and evaluation
data_channels = {"train": fs_train_key, "test": fs_test_key}
estimator.fit(inputs=data_channels, wait=False)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)




# Model Endpoint predictor API
class DeepARPredictor(sage.predictor.Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            serializer=IdentitySerializer(content_type="application/json"),  # for robust class serialization
            **kwargs,
        )

    def Predict(self, fs, cat=None, dynamic_feat=None, num_samples=50, return_samples=False, quantiles=None):
        if quantiles is None:
            quantiles = ["0.1", "0.5", "0.9"]
        prediction_time = fs.index[-1] + fs.index.freq
        quantiles = [str(q) for q in quantiles]
        req = self.__encode_request(fs, cat, dynamic_feat, num_samples, return_samples, quantiles)
        res = super(DeepARPredictor, self).predict(req)
        return self.__decode_response(res, fs.index.freq, prediction_time, return_samples)

    def __encode_request(self, fs, cat, dynamic_feat, num_samples, return_samples, quantiles):
        instance = series_to_dict(
            fs, cat if cat is not None else None, dynamic_feat if dynamic_feat else None
        )

        configuration = {
            "num_samples": num_samples,
            "output_types": ["quantiles", "samples"] if return_samples else ["quantiles"],
            "quantiles": quantiles,
        }

        http_request_data = {"instances": [instance], "configuration": configuration}

        return json.dumps(http_request_data).encode("utf-8")

    def __decode_response(self, response, freq, prediction_time, return_samples):
        predictions = json.loads(response.decode("utf-8"))["predictions"][0]
        prediction_length = len(next(iter(predictions["quantiles"].values())))
        prediction_index = pd.date_range(
            start=prediction_time, freq=freq, periods=prediction_length
        )
        if return_samples:
            dict_of_samples = {"sample_" + str(i): s for i, s in enumerate(predictions["samples"])}
        else:
            dict_of_samples = {}
        return pd.DataFrame(
            data={**predictions["quantiles"], **dict_of_samples}, index=prediction_index
        )

    def set_frequency(self, freq):
        self.freq = freq


def encode_target(fs):
    return [x if np.isfinite(x) else "NaN" for x in fs]


def series_to_dict(fs, cat=None, dynamic_feat=None):
    obj = {"start": str(fs.index[0]), "target": encode_target(fs)}
    if cat is not None:
        obj["cat"] = cat
    if dynamic_feat is not None:
        obj["dynamic_feat"] = dynamic_feat
    return obj

# Model production Endpoint predictor
endpoint_name = f'FoodSecurity-AR-{strftime("%Y-%m-%d-%H-%M-%S"), gmtime()}'
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    predictor_cls=DeepARPredictor,
    endpoint_name=endpoint_name,
)

# Lambda handler to read input from S3 and invoke SageMaker endpoint
def lambda_handler(event, context):
    # Initialize a SageMaker runtime client
    runtime_client = bt3.client('runtime.sagemaker')

    # Initialize an S3 client
    s3_client = bt3.client('s3')

    # Extract bucket name and key from the event
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Fetch the input data from S3
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        input_data = response['Body'].read().decode('utf-8')
    except Exception as e:
        return {
            "statusCode": 400,
            "
"""
