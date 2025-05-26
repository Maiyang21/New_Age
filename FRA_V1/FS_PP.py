# importing used libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as MICE

""" Data preparation for model ingestion """

# Data importation
FS_Feat = pd.read_csv("C:/Users/PC/Documents/DATABASE/team datalink/FS_FeatureStore/FS_V1.csv")
features = ['Availability Metric', 'Average Caloric Value', 'Average Person Calories', 'Population', 'Value_AVAIL',
            'AccessibilityMetric', 'Value_P', 'Value_PPP', 'UtilizationMetric', 'Value_SUA']


# Data pivoting for feature cleaning
def data_pivot(ds: pd.DataFrame, features: list) -> pd.DataFrame:
    ds['Year'] = features
    ds = pd.pivot_table(data=ds, columns=ds['Year'])
    return ds

"""
# Handling missing values
def data_prep(ds: pd.DataFrame) -> pd.DataFrame:
    missing_vals_list = list(ds.isnull().sum())
    column_vals_list = [len(ds[i].values) for i in ds.columns]

    for i, j in zip(missing_vals_list, column_vals_list):
        if i / j <= 0.1:
            ds = ds.dropna()
        elif 0.1 < i / j <= 0.5:
            ds = ds.fillna(ds.mean())
        elif i / j > 0.5:
            fs_mice = MICE(max_iter=10, random_state=20)
            ds = fs_mice.fit_transform(ds)

    return ds
"""

# test run
print(data_pivot(FS_Feat, features).head(5))
# final_frame= data_prep(data_pivot(FS_Feat, features))
# print(data_pivot(final_frame, final_frame['Year']).head(5))

# implementing data preprocessing
FS_Feat_prep = data_pivot(FS_Feat, features)
FS_Feat_prep.to_csv("C:/Users/PC/Documents/DATABASE/team datalink/FS_FeatureStore/FS_V1_prep.csv", index=bool(1))