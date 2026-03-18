import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

CITIES = [
    '三亚', '上海', '上饶', '东莞', '中山', '临沂', '丹东', '乌鲁木齐', '九江', '佛山',
    '保定', '兰州', '凉山', '包头', '北京', '北海', '南京', '南充', '南宁', '南昌',
    '南通', '厦门', '台州', '合肥', '吉安', '吉林', '周口', '呼和浩特', '咸阳', '哈尔滨',
    '唐山', '嘉兴', '大连', '天水', '天津', '太原', '威海', '宁波', '安庆', '宜宾',
    '宜昌', '宝鸡', '岳阳', '常州', '常德', '平顶山', '广元', '广州', '廊坊', '开封',
    '张家口', '徐州', '德阳', '惠州', '成都', '承德', '抚顺', '攀枝花', '新乡', '无锡',
    '昆明', '晋中', '杭州', '柳州', '株洲', '桂林', '武汉', '汉中', '江门', '沈阳',
    '泉州', '泰安', '洛阳', '济南', '济宁', '海口', '淄博', '淮安', '深圳', '清远',
    '温州', '湖州', '湘西', '湛江', '漳州', '潍坊', '濮阳', '烟台', '珠海', '盐城',
    '石家庄', '福州', '绍兴', '绵阳', '芜湖', '苏州', '菏泽', '衡阳', '衢州', '襄阳',
    '西安', '许昌', '贵阳', '资阳', '赣州', '赤峰', '达州', '运城', '通辽', '遂宁',
    '邯郸', '郑州', '鄂州', '重庆', '金华', '银川', '镇江', '长春', '长沙', '阜阳',
    '防城港', '雅安', '青岛', '马鞍山', '驻马店', '黄石'
]

# ⚠️ 重要：根据你的显存大小决定同时跑几个。
# 如果显存很大或任务显存占用小，可以设高点；否则设为 2-4。
MAX_PARALLEL_TASKS = 3

# 如果有多个显卡，可以列出索引，例如 [0, 1]
GPU_IDS = [0] 

def run_city_task(city, gpu_id):
    """单个城市的处理任务"""
    print(f"🚀 启动城市: {city} (分配至 GPU: {gpu_id})")
    
    # 设置当前进程的环境变量，指定显卡
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        # 使用 subprocess.run 执行，并传入环境变量
        result = subprocess.run(
            [sys.executable, 'run_all.py', city],
            capture_output=False,
            env=env
        )
        
        if result.returncode == 0:
            return True, city
        else:
            return False, city
    except Exception as e:
        return False, f"{city} (Error: {str(e)})"

def main():
    failed_cities = []
    success_count = 0
    
    print(f"🔥 开始并行处理，最大并行数: {MAX_PARALLEL_TASKS}")
    
    # 使用进程池
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_TASKS) as executor:
        # 提交任务
        # 这里简单地循环分配 GPU ID
        futures = {
            executor.submit(run_city_task, city, GPU_IDS[i % len(GPU_IDS)]): city 
            for i, city in enumerate(CITIES)
        }
        
        # 实时获取完成结果
        for future in as_completed(futures):
            success, city_name = future.result()
            if success:
                print(f"✅ 城市 {city_name} 处理完成")
                success_count += 1
            else:
                print(f"❌ 城市 {city_name} 处理失败")
                failed_cities.append(city_name)

    # 总结汇报
    print(f"\n{'='*50}")
    print(f"任务结束！成功: {success_count}, 失败: {len(failed_cities)}")
    if failed_cities:
        print(f"失败列表: {', '.join(failed_cities)}")
    print('='*50)

if __name__ == '__main__':
    main()