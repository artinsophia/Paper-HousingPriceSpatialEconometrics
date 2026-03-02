import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import os

def Display(df):
    print(f"Shape: {df.shape}")
    print(df.columns)
    missing_counts = df.isnull().sum()
    unique_counts = df.nunique()
    summary_df = pd.DataFrame({
        'Missing': missing_counts,
        'Unique': unique_counts,
        'Dtype': df.dtypes
    })
    display(summary_df)

    for col in df.columns:
        print(f"【col: {col}】")
        top_5 = df[col].value_counts().head(5)
        
        if top_5.empty:
            print("  (该列全为空或无有效数据)")
        else:
            for val, count in top_5.items():
                print(f"  - {val}: {count} 次")
        print("-" * 40)

def data_cleaning(park,housing):
    park_cleaned = park.dropna(subset=['经度','纬度'])
    housing_cleaned = housing.dropna(subset=['lon','lat','单价'])
    housing_cleaned = housing_cleaned.rename(columns={'lon':'x','lat':'y','单价':'price'})
    park_cleaned = park_cleaned.rename(columns={'经度':'x','纬度':'y'})
    sub_housing = housing_cleaned.drop(columns=['环线','套内面积','抵押信息']).copy()
    sub_park = park_cleaned[['x', 'y','产业']].copy()
    return sub_park,sub_housing

