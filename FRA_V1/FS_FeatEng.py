# libraries to be used
import pandas as pd
import numpy as np

"Feature Engineering of the Food Security Dataset"

# Dataset importation
ds_avail = pd.read_csv("C:/Users/PC/Documents/DATABASE/team datalink/Availability_DB/Availability_etl.csv")
ds_access = pd.read_csv("C:/Users/PC/Documents/DATABASE/team datalink/Accessibility_DB/Accessibility_etl.csv")
ds_util = pd.read_csv("C:/Users/PC/Documents/DATABASE/team datalink/Utilization_DB/Utility_etl.csv")


# ds_util= pd.read_csv("C:/Users/PC/Documents/DATABASE/team datalink/Utility_DB/Utility_etl.csv")


# Availability Metric
def Avail_feat(ds: pd.DataFrame) -> pd.DataFrame:
    ds['Availability Metric'] = ds['ItemVolumetricCalories'] - ds['ConsumptionCaloriesDensity']
    for i in ds.columns:
        if i not in ['Availability Metric', 'Year', 'Population', 'Value_AVAIL', 'Average Caloric Value',
                     'Average Person Calories']:
            ds = ds.drop([i], axis=1)
    for i in ds.columns:
        ds[i] = ds[i].values.astype(int)
    ds = pd.pivot_table(data=ds, columns=ds['Year'])
    return ds


# run test
# print(Avail_feat(ds_avail).head(5))


# Accessibility Metric
def Access_feat(ds: pd.DataFrame) -> pd.DataFrame:
    for i in ds.columns:
        if i not in ['AccessibilityMetric', 'Year', 'Value_PPP', 'Value_P',
                     'Average Person Calories']:
            ds = ds.drop([i], axis=1)
    ds = pd.pivot_table(data=ds, columns=ds['Year'])
    return ds


# test run
# print(Access_feat(ds_access).head(5))


# Utility Metric
def Util_feat(ds: pd.DataFrame) -> pd.DataFrame:
    for i in ds.columns:
        if i not in ['Year', 'UtilizationMetric', 'Value_SUA']:
            ds = ds.drop([i], axis=1)
    ds = pd.pivot_table(data=ds, columns='Year')
    return ds


# test run
# print(Util_feat(ds_util).head(5))


# unified feature sets
def fs_feat(ds1: pd.DataFrame, ds2: pd.DataFrame, ds3: pd.DataFrame) -> pd.DataFrame:
    ds = pd.concat([ds1, ds2, ds3])
    return ds


# test run
print(fs_feat(Avail_feat(ds_avail), Access_feat(ds_access), Util_feat(ds_util)))


# exporting Feature dataset to database
FS_feat = fs_feat(Avail_feat(ds_avail), Access_feat(ds_access), Util_feat(ds_util))
FS_feat.to_csv("C:/Users/PC/Documents/DATABASE/team datalink/FS_FeatureStore/FS_V1.csv", index=bool(0))
