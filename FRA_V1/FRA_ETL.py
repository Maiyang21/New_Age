import pandas as pd
import pymysql, csv, os
import kaggle
import kaggledatasets
from mysql.connector import connection
import numpy as np


def check_ava_data() -> list:
    res = kaggle.api.dataset_list(search='finance loan data')
    return res


def fetch_new_data():
    ls = kaggle.api.datasets_list(search='finance loan data')
    ls_df = pd.DataFrame(ls)
    for i in ls_df['ref']:
        if i not in check_ava_data():
            kaggle.api.dataset_download_files(i, path="C:/Users/PC/Documents/DATABASE", quiet=bool(0), unzip=bool(1))
            check_ava_data().append(i)


def connect_DBMS():
    conn = pymysql.connect(host='127.0.0.1', user='root', password='Domination#21', database='fra_analysis')
    cursor = conn.cursor()
    res = cursor.execute("SELECT * FROM fra_analysis.fra_dropped;")
    rows = cursor.fetchall()
    return rows


def load_DBMS():
    kaggle_list = []
    for i in os.listdir("C:/Users/PC/Documents/DATABASE/kaggle"):
        if i not in kaggle_list:
            path = os.path.join("C:/Users/PC/Documents/DATABASE/kaggle", i)
            new = csv.reader(path)

    new_data = csv.reader("C:/Users/PC/Documents/DATABASE/train.csv")


print(connect_DBMS())
