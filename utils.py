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
    sub_housing = housing_cleaned.drop(columns=['环线','套内面积','抵押信息','链家编号','小区名称','房本备件','建筑面积']).copy()
    sub_park = park_cleaned[['x', 'y','产业','级别','产业园名称','省份','城市','区县']].copy()
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

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  
    return km

def multi_label_explosion(df, col_name, split):

    unique_types = (
        df[col_name]
        .dropna()
        .str.split(split)
        .explode()
        .str.strip()  
        .unique()
    )

    print(f"检测到的基础{col_name}类型共有 {len(unique_types)} 种：")
    print(unique_types)
    
    for t in unique_types:
        df[f'{col_name}_is_{t}'] = df[col_name].str.contains(t, na=False, regex=False).astype(int)

    return df

def cn_to_int_custom(cn_str):
    """纯Python实现：将中文数字（含百、十、零、两）转为整数"""
    if not cn_str:
        return 0
    
    # 基础映射
    digits = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, 
              '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    units = {'十': 10, '百': 100}
    
    total = 0
    temp_val = 0  # 存储当前数字，待与单位相乘
    
    for char in cn_str:
        if char in digits:
            temp_val = digits[char]
        elif char in units:
            unit_val = units[char]
            # 处理“十一”这种省略开头“一”的情况
            if temp_val == 0 and char == '十':
                temp_val = 1
            total += temp_val * unit_val
            temp_val = 0
        elif char == '零':
            continue
            
    total += temp_val # 加上末尾的个位数
    return total


