import papermill as pm
import sys
import os

if len(sys.argv) < 2:
    print("用法: python run_all.py <city>")
    sys.exit(1)

city = sys.argv[1]

log_dir = f"logs/run_logs_{city}"
os.makedirs(log_dir, exist_ok=True)


notebooks = [
    "01_data_cleaning.ipynb",
    "02_spatial_encoding.ipynb",
    "03_feature_engineering.ipynb",
    "04_statistical_analysis.ipynb",
    "05_EDA.ipynb"
]

print(f"开始处理城市: {city}")
print("-" * 30)

for notebook in notebooks:
    input_path = notebook
    output_path = os.path.join(log_dir, f"result_{notebook}")
    
    print(f"正在执行: {notebook} ...", end=" ", flush=True)
    
    try:
        pm.execute_notebook(
            input_path,
            output_path,
            parameters=dict(city=city),
            log_output=False, 
            progress_bar=False 
        )
    except Exception as e:
        print(f"错误信息: {e}")
        sys.exit(1) 

print("-" * 30)
print(f"所有任务已执行完毕！结果保存在: {log_dir}")