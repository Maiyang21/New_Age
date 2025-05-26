# Required Libraries
import os
import json
import pandas as pd
import numpy as np
import boto3 as bt3
import sagemaker as sage
from sagemaker.serializers import IdentitySerializer
from datetime import datetime
from time import strftime, gmtime
from sagemaker.serializers import IdentitySerializer, JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
from io import BytesIO

# Initialize SageMaker session
sagemaker_session = sage.Session()
role = sage.get_execution_role()

"""
# Load Parquet Data for Inference
def load_parquet(file_path):
    df = pd.read_parquet(file_path)
    return df


# Convert DataFrame Series to Model Input Format
def encode_target(fs):
    return [x if np.isfinite(x) else "NaN" for x in fs]


def series_to_dict(fs, cat=None, dynamic_feat=None):
    obj = {"start": str(fs.index[0]), "target": encode_target(fs)}
    if cat is not None:
        obj["cat"] = cat
    if dynamic_feat is not None:
        obj["dynamic_feat"] = dynamic_feat
    return obj
"""


# DeepAR Predictor Class

class DeepARPredictor(Predictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super().__init__(
            endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )

    def set_prediction_parameters(self, freq, prediction_length):
        """Set frequency and prediction length parameters."""
        self.freq = freq
        self.prediction_length = prediction_length

    def predict(self, ts, cat=None, dynamic_feat=None,
                encoding="utf-8", num_samples=100, quantiles=["0.1", "0.5", "0.9"]):
        """Requests predictions from a DeepAR endpoint.

        Args:
            ts (np.ndarray or pd.Series): Target time series data. Assumes the FIRST element corresponds to the 'start' time.
            cat (int, optional): Category index. Defaults to None.
            dynamic_feat (np.ndarray, optional): Dynamic features. Defaults to None.
            encoding (str): Encoding for the request. Defaults to "utf-8".
            num_samples (int): Number of sample paths to generate. Defaults to 100.
            quantiles (list[str]): Quantiles to predict. Defaults to ["0.1", "0.5", "0.9"].

        Returns:
            dict: Dictionary containing prediction results ('quantiles' and potentially 'samples').
        """
        # Convert pandas Series to list if necessary
        target_list = ts.tolist() if isinstance(ts, pd.Series) else ts

        # Determine start date string from the index if available
        if isinstance(ts, pd.Series) and isinstance(ts.index, pd.DatetimeIndex):
            # Format the first timestamp as required 'YYYY-MM-DD HH:MM:SS'
            start_dt_str = ts.index[0].strftime('%Y-%m-%d %H:%M:%S')
        else:
            # Using a placeholder:
            start_dt_str = "1970-01-01 00:00:00"
            print("Warning: Time series 'start' date could not be determined from index. Using placeholder.")

        # Construct the request payload matching the standard DeepAR format
        instances = []
        instance = {"start": start_dt_str,
                    "target": target_list}
        if cat is not None:
            instance["cat"] = cat
        if dynamic_feat is not None:
            # Ensure dynamic_feat has the correct shape: (num_features, length)
            # Length needs to cover context_length + prediction_length for inference
            instance["dynamic_feat"] = dynamic_feat.tolist()
        instances.append(instance)

        configuration = {
            "num_samples": num_samples,
            "output_types": ["quantiles", "samples"],  # Request both
            "quantiles": quantiles
        }

        request_data = {"instances": instances, "configuration": configuration}

        # Make the prediction request using the parent class's predict method
        predictions = super().predict(request_data)

        return predictions


# Helper function to format predictions into a DataFrame
def decode_predictions(predictions, freq, start_date):
    """Transforms DeepAR prediction results into a Pandas DataFrame.

    Args:
        predictions (dict): The dictionary returned by the DeepAR endpoint.
        freq (str): The frequency of the time series (e.g., 'M', 'D').
        start_date (pd.Timestamp): The timestamp of the first prediction.

    Returns:
        pd.DataFrame: A DataFrame containing predictions, indexed by time.
    """
    # We only look at the first instance's predictions
    # Check if predictions are structured as expected
    if not predictions or 'predictions' not in predictions or not predictions['predictions']:
        raise ValueError("Invalid prediction response format.")

    prediction_results = predictions['predictions'][0]
    if 'quantiles' not in prediction_results or '0.5' not in prediction_results['quantiles']:
        raise ValueError("Prediction response missing expected 'quantiles' structure.")

    prediction_length = len(prediction_results['quantiles']['0.5'])  # Use median length
    prediction_index = pd.date_range(start=start_date, freq=freq, periods=prediction_length)

    # Extract quantiles
    quantile_data = prediction_results['quantiles']

    # Extract samples if available
    sample_data = {}
    if 'samples' in prediction_results:
        samples = prediction_results['samples']
        for i, sample in enumerate(samples):
            sample_data[f'sample_{i}'] = sample

    # Combine into DataFrame
    df_data = {**quantile_data, **sample_data}
    return pd.DataFrame(df_data, index=prediction_index)


def deploy_trained_model(estimator_to_deploy, initial_instance_count=1, instance_type="ml.m5.large"):
    """ Deploys the endpoint using the estimator that ran the training job. """
    print(f"Deploying model from training job: {estimator_to_deploy.latest_training_job.job_name}")

    # Generate a unique endpoint name
    endpoint_name = f"deepar-FoodSecurity-endpoint-{strftime('%Y-%m-%d-%H-%M-%S', gmtime())}"
    print(f"Creating endpoint: {endpoint_name}")

    # Deploy using the original estimator and specifying standard JSON handling
    predictor = estimator_to_deploy.deploy(
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )

    print(f"Endpoint {endpoint_name} created successfully.")

    # Now, wrap the deployed endpoint name with our custom predictor class for convenience methods
    custom_predictor = DeepARPredictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker_session)
    # Set prediction parameters if needed (though they are passed during predict call now)
    # custom_predictor.set_prediction_parameters(freq, prediction_length)

    return custom_predictor, endpoint_name


# Inference function
def run_inference(predictor, target_ts, category=None, dynamic_features=None):
    """Runs inference using the deployed predictor.

    Args:
        predictor (DeepARPredictor): The predictor instance.
        target_ts (pd.Series): The input time series with a DatetimeIndex.
        category (int, optional): Categorical feature ID. Defaults to None.
        dynamic_features (np.ndarray, optional): Dynamic features array. Defaults to None.

    Returns:
        pd.DataFrame or dict: DataFrame of predictions or raw dictionary on error.
    """

    if not isinstance(target_ts.index, pd.DatetimeIndex):
        raise ValueError("Input time series must have a DatetimeIndex.")

    print(f"Running inference for time series starting {target_ts.index[0]}...")

    # custom predictor's predict method which handles payload formatting
    prediction_results = predictor.predict(
        ts=target_ts,
        cat=category,
        dynamic_feat=dynamic_features
        # num_samples and quantiles use defaults defined in the class method
    )
    # Infer frequency from the input series if possible
    freq= target_ts.index.freq
    inferred_freq = pd.infer_freq(target_ts.index)
    if not inferred_freq:
        print(f"Warning: Could not infer frequency from input series. Using base frequency '{freq}'.")
        current_freq = freq  # Use the global freq defined earlier
    else:
        current_freq = inferred_freq

    # Calculating prediction start date based on the frequency
    if current_freq in ('M', 'MS'):
        prediction_start_date = target_ts.index[-1] + pd.offsets.MonthEnd(1)
    else:
        offset = pd.tseries.frequencies.to_offset(current_freq)
        if offset:
            prediction_start_date = target_ts.index[-1] + offset
        else:  # A fallback if offset cannot be determined easily
            print(
                f"Warning: Cannot reliably determine offset for freq '{current_freq}'. Prediction index might be inaccurate.")
            # Attempting to use the last date + 1 day as a fallback
            prediction_start_date = target_ts.index[-1] + pd.Timedelta(days=1)

    print(f"Decoding predictions starting from: {prediction_start_date}")
    try:
        predictions_df = decode_predictions(prediction_results, current_freq, prediction_start_date)
        return predictions_df
    except ValueError as e:
        print(f"Error decoding predictions: {e}")
        print("Returning raw prediction results.")
        return prediction_results  # Returning raw results if decoding fails


"""
# Main Execution
if __name__ == "__main__":
    file_path = "your_parquet_file.parquet"  # Update with actual file path
    df = load_parquet(file_path)

    predictor, endpoint_name = deploy_model()
    predictions = run_inference(predictor, df)

    print(predictions)
"""
