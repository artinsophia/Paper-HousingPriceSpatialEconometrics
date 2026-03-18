import os
import pandas as pd
from dotenv import load_dotenv
import qianfan

city = "北京"

import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import time

# 1. 初始化
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)

def get_park_year(park_name, province, city):
    """单条查询函数"""
    prompt = f"请帮我查询位于{province}{city}的‘{park_name}’的建造、动工或落成时间。要求：仅返回年份（如 2010）或日期（如 2010-05），不要解释，查不到返回'Unknown'。"
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个精准的数据抓取助手，只输出核心结果。"},
                {"role": "user", "content": prompt}
            ],
            timeout=15 # 设置超时，防止死等
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    input_file = f"processed_data/{city}/01_park.csv"
    output_file = f"processed_data/park/02_park.csv"

    # 读取数据
    df = pd.read_csv(input_file)
    if '建造时间' not in df.columns:
        df['建造时间'] = None

    print(f"--- 任务启动：共 {len(df)} 条数据 ---")
    
    # 设置运行限制（测试时设为 5，正式跑设为 len(df)）
    limit = len(df)  # 正式跑
    count = 0

    for index, row in df.iterrows():
        # 如果已经有数据了，跳过（断点续传）
        if pd.notna(row['建造时间']) and "Error" not in str(row['建造时间']):
            continue
        
        # 达到测试上限就退出
        if count >= limit:
            print(f"\n📢 已完成测试模式（前 {limit} 条）。请检查生成的 {output_file}，满意后再修改 limit 继续跑。")
            break

        # 执行查询
        park_name = row['产业园名称']
        res = get_park_year(park_name, row['省份'], row['城市'])
        
        # --- 实时反馈：在这里你可以亲眼看到结果 ---
        print(f"[{index+1}/{len(df)}] 正在查询: {park_name[:15]}... \t 结果: {res}")
        
        # 更新数据
        df.at[index, '建造时间'] = res
        count += 1

        # 每 5 条保存一次文件，防止意外
        if count % 5 == 0:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"   💾 已自动保存进度到 {output_file}")

        # 适当停顿，避免被封（DeepSeek 建议加个小延迟）
        time.sleep(0.5)

    # 任务结束保存
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("\n✅ 处理圆满结束！")

if __name__ == "__main__":
    main()