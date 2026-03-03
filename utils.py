import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import os
import re

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
    housing_cleaned = housing.dropna(subset=['lon','lat','单价','价格'])
    housing_cleaned = housing_cleaned.rename(columns={'lon':'x','lat':'y','单价':'unit_price','价格':'price'})
    park_cleaned = park_cleaned.rename(columns={'经度':'x','纬度':'y'})
    sub_housing = housing_cleaned.drop(columns=['环线','套内面积','抵押信息','链家编号','看房时间']).copy()
    sub_park = park_cleaned[['x', 'y','产业']].copy()
    return sub_park,sub_housing

def process_num(text):

    if pd.isna(text):
        return None
    nums = re.findall(r"\d+\.?\d*", text)
    nums = [float(n) for n in nums]

    if len(nums) >= 2:
        avg = sum(nums[:2]) / 2
        return avg
    elif len(nums) == 1:
        return nums[0]
    else:
        return None
    
def extract_year(text):
    if pd.isna(text):
        return None
    match = re.search(r'(\d{4})', str(text))
    if match:
        return int(match.group(1))
    return None

def extract_year(text):
    if pd.isna(text):
        return None
    match = re.search(r'(\d{4})', str(text))
    if match:
        return int(match.group(1))
    return None

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  
    return km

def smooth_target_encoding(train, test, column, target, weight=10):

    # 1. 计算全域平均值 (Global Mean)
    global_mean = train[target].mean()
    
    # 2. 计算每个板块的统计量：均值和计数
    agg = train.groupby(column)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    
    # 3. 计算平滑权重 alpha
    smooth_weights = counts / (counts + weight)
    
    # 4. 计算编码值
    encoded_values = smooth_weights * means + (1 - smooth_weights) * global_mean
    
    # 5. 映射回原表
    train_encoded = train[column].map(encoded_values)
    test_encoded = test[column].map(encoded_values).fillna(global_mean) # 测试集若有新板块，填入全局均值
    
    return train_encoded, test_encoded