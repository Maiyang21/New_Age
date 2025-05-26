# libraries to be used
import os
import json
import pandas as pd
import numpy as np
from datetime import timedelta
import boto3 as bt3
import sagemaker as sage
from sagemaker.serializers import IdentitySerializer, JSONSerializer
from time import get_clock_info, strftime

"""MODEL BUILD WITH AWS CLOUD"""

# Environment setup,S3 bucket prep and session creation
fs_region = bt3.session.Session().region_name
s3 = bt3.resource('s3')
bucket = 'fs_bucket'
role = sage.get_execution_role()  # IAM role for sagemaker

try:
    if fs_region == 'us-east-1':
        s3.create_bucket(bucket_name=bucket)
        print('bucket successfully created')

except Exception as e:
    print('bucket creation failed:', e)

# bucket object key creation
output_prefix = 'FS_METRIC'
fs_out_key = 's3://{}/{}/out'.format(bucket, output_prefix)
input_1_prefix = 'TRAIN_METRIC'
fs_train_key = 's3://{}/{}/training'.format(bucket, input_1_prefix)
input_2_prefix = 'TRAIN_METRIC'
fs_test_key = 's3://{}/{}/testing'.format(bucket, input_2_prefix)

# data formating
fs_train = pd.read_csv('C:/Users/PC/Documents/DATABASE/team datalink/FS_FeatureStore/FS_V1_prep.csv')
freq = 'Y'
prediction_length = 10  # 10 year
context_length = fs_train['Year'].max() - fs_train['Year'].min()
general_start = fs_train['Year'].min()
general_end = fs_train['Year'].max()

Train_data ={
    "start": str(general_start),
    "target": fs_train['Availability Metric'][general_start:general_end - prediction_length].tolist(),
    "dynamic_feat": [
        fs_train['Average Caloric Value'][general_start:general_end - prediction_length].tolist(),
        fs_train['Average Person Calories'][general_start:general_end - prediction_length].tolist(),
        fs_train['Population'][general_start:general_end - prediction_length].tolist(),
        fs_train['Value_AVAIL'][general_start:general_end - prediction_length].tolist(),
        fs_train['AccessibilityMetric'][general_start:general_end - prediction_length].tolist(),
        fs_train['Value_P'][general_start:general_end - prediction_length].tolist(),
        fs_train['Value_PPP'][general_start:general_end - prediction_length].tolist(),
        fs_train['UtilizationMetric'][general_start:general_end - prediction_length].tolist(),
        fs_train['Value_SUA'][general_start:general_end - prediction_length].tolist()
    ]
}

Test_data = {
    "start": str(general_start),
    "target": fs_train['Availability Metric'],
    "dynamic_feat": [fs_train['Average Caloric Value'], fs_train['Average Person Calories'], fs_train['Population'],
                     fs_train['Value_AVAIL'], fs_train['AccessibilityMetric'], fs_train['Value_P'],
                     fs_train['Value_PPP'],
                     fs_train['UtilizationMetric'], fs_train['Value_SUA']]
}


# function to converting json dict format to file
def write_dicts_to_file(path, data):
    with open(path, "wb") as f:
        for d in data:
            f.write(json.dumps(d).encode("utf-8"))
            f.write("\n".encode('utf-8'))


# implementing json conversion function
write_dicts_to_file("C:/Users/PC/Documents/DATABASE/team datalink/FS_model_formats/fs_train_v1.json", Train_data)
write_dicts_to_file("C:/Users/PC/Documents/DATABASE/team datalink/FS_model_formats/fs_test_v1.json", Test_data)


# function to Upload preprocessed data to cloud
def S3_upload(file, path, override=bool(0)):
    split = path.split('/')
    buck = split[2]
    key = '/'.join(split[3:])
    buk = s3.Bucket(bucket)

    if len(list(buk.objects.filter(Prefix=key))) > 0:
        if not override:
            print(
                "File s3://{}/{} already exists.\nSet override to upload anyway.\n".format(
                    buck, path
                )
            )
            return
        else:
            print("Overwriting existing file")
    with open(file, "rb") as data:
        print("Uploading file to {}".format(path))
        buk.put_object(Key=key, Body=data)


# implementing json cloud upload
S3_upload("C:/Users/PC/Documents/DATABASE/team datalink/FS_model_formats/fs_train_v1.json",
          fs_train_key + "/train.json")
S3_upload("C:/Users/PC/Documents/DATABASE/team datalink/FS_model_formats/fs_test_v1.json", fs_test_key + "/test.json")

# Model initialization
image_name = sage.container_def(image_uri='forcasting-deepar')  # container call on global
sagemaker_session = sage.Session()

estimator = sage.estimator.Estimator(
    image_uri='forcasting-deepar',
    sagemaker_session=sagemaker_session,
    role=role,
    train_instance_count=1,
    train_instance_type="ml.c4.2xlarge",
    base_job_name="deepar-FoodSecurity-v1",
    output_path=fs_out_key,
    use_spot_instances=bool(1)
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
estimator.fit(inputs=data_channels, wait=bool(1))

# Inference job test initialization and endpoint definition
FS_job = estimator.latest_training_job.name

s_endpoint_name = sagemaker_session.endpoint_from_job(
    job_name=FS_job,
    initial_instance_count=1,
    instance_type="ml.c5.large",
    image_uri='forcasting-deepar',
    role=role,
)

# Model staging test predictor
s_predictor = sage.predictor.Predictor(
    endpoint_name=s_endpoint_name, sagemaker_session=sagemaker_session, serializer=JSONSerializer()
)



# Model Endpoint predictor API
class DeepARPredictor(sage.predictor.Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            # serializer=JSONSerializer(),
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
endpoint_name = 'FoodSecurity-AR' + strftime("%Y-%M-%H-%M-%S".get_clock_info())
predictor = estimator.deploy(
    initial_instance_count=1, instance_type="ml.m5.large", predictor_cls=DeepARPredictor, endpoint_name=endpoint_name,
)

# Invoking a lambda function for API creation
# Initialize a SageMaker runtime client
runtime_client = bt3.client('runtime.sagemaker')

# Define the S3 client
s3_client = bt3.client('s3')


def lambda_handler(event, context):
    # Extract bucket name and key from the event
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Fetch the input data from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    input_data = response['Body'].read().decode('utf-8')

    # Convert the input data to JSON
    sample = json.loads(input_data)

    # Define the endpoint name
    endpoint_name = 'Food-Security-AR'

    # Invoke the SageMaker endpoint
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(sample)
    )

    # Get the prediction result
    result = json.loads(response['Body'].read().decode('utf-8'))

    # Return the prediction result
    return {
        "statusCode": 200,
        "body": json.dumps({"prediction": result})
    }



def lambda_handle_req(event, context):
    runtime_client = bt3.client('runtime.sagemaker')
    endpoint_name = 'Food-Security-AR'
    sample = json.loads(json.dumps(event))

    response = runtime_client.invoke_endpoint(EndPointName=endpoint_name, ContentType='application/json',
                                              Body=str(sample['body']))
    
    result = json.loads(response['Body'].read().decode('utf-8'))

    return {"statusCode": 200,
            "body": json.dumps({"prediction": result})}

# STABILITY METRIC
# def Stability_Metric(fs1,fs2) -> int:
"""
