# libraries to be used
import pandas as pd
import numpy as np

# Data ingestion
t_av = pd.read_csv(
    "C:/Users/PC/Documents/DATABASE/team datalink/Availability_DB/avail_metric/updated_population_dataset.csv")
t_cc = pd.read_csv(
    "C:/Users/PC/Documents/DATABASE/team datalink/Availability_DB/avail_metric/average_nigerian_consumption_calories.csv")
t_ac = pd.read_csv("C:/Users/PC/Documents/DATABASE/team datalink/Accessibility_DB/producer_prices.csv")
t_pp = pd.read_csv("C:/Users/PC/Documents/DATABASE/team datalink/Accessibility_DB/access_metric/Nigeria_PPP.csv")
t_ut = pd.read_csv("C:/Users/PC/Documents/DATABASE/team datalink/Utilization_DB/food__class_availability.csv")

"""   Data cataloging and ETL for Availability metrics """


def data_avail_ingest(ds: pd.DataFrame, ds2: pd.DataFrame) -> pd.DataFrame:
    # inner joining  item caloric values
    ds1 = pd.merge(ds, ds2, on='Item', how='left')
    ds1['AvgCalories'] = ds1['Average Caloric Value']
    # cleaning population data
    ds1["Population"] = ds1["Average Population (millions)"]
    for i in range(len(ds1['Population'])):
        if i in [0, 1, 2, 3]:
            ds1['Population'][i] = ds1['Population'][5].astype(int)
        elif i % 5 == 0:
            ds1['Population'][i] = ds1['Population'][i].astype(int)
        else:
            ds1['Population'][i] = ds1['Population'][i - 1].astype(int)
    # generating the availability metric
    ds1['Average Person Calories'] = ds1['Avg Nigerian Consumption (Calories/Year)']
    ds1['ItemVolumetricCalories'] = ds1['Value'] * ds1['AvgCalories']
    ds1['ConsumptionCaloriesDensity'] = ds1['Population'] * ds1['Average Person Calories']
    ds1= ds1.rename(columns={'Value':'Value_AVAIL'})
    return ds1


# function testing
# print(data_avail_ingest(t_av, t_cc))

# loading Avail data to env
Avail_load = data_avail_ingest(t_av, t_cc)
Avail_load.to_csv("C:/Users/PC/Documents/DATABASE/team datalink/Availability_DB/Availability_etl.csv", index=bool(0))


def data_access_ingest(ds1: pd.DataFrame, ds2: pd.DataFrame) -> pd.DataFrame:
    # Merging data sets
    for i in ds1['Element']:
        if i in ds1['Element'] != "Producer Price Index (2014-2016 = 100)":
            ds1['Element'] = ds1["Element"].drop([i], axis=1)
    ds1['Value_PPP'] = ds1['Value'].values.astype(int)
    ds2['Value_P'] = ds2['Value'].values.astype(int)
    ds = pd.merge(ds1, ds2, on='Year', how='left')
    # generating the accessibility metric
    ds['AccessibilityMetric'] = ds['Value_PPP'] / ds['Value_P']

    return ds


# function testing
# print(data_access_ingest(t_ac, t_pp))

# loading access data to ETL_dB
Access_load = data_access_ingest(t_ac, t_pp)
Access_load.to_csv("C:/Users/PC/Documents/DATABASE/team datalink/Accessibility_DB/Accessibility_etl.csv", index=bool(0))


def data_util_ingest(ds1: pd.DataFrame, ds2: pd.DataFrame) -> pd.DataFrame:
    # Generating the utilization metric
    ds1['UtilizationMetric'] = ds1['Value'].values.astype(int)
    ds1= ds1.rename(columns={'Value':'Value_SUA'})

    return ds1


# function testing
# print(data_util_ingest(t_ut, t_cc))

# loading util data into etl_db
Util_load = data_util_ingest(t_ut, t_cc)
Util_load.to_csv("C:/Users/PC/Documents/DATABASE/team datalink/Utilization_DB/Utility_etl.csv", index=bool(0))
