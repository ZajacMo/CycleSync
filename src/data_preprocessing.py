import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 配置
DATA_PATH = "题C-附件-mobike_shanghai_dataset.csv"
OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 读取数据
print("Loading data...")
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    print(f"Error reading CSV: {e}")
    # 尝试更鲁棒的读取方式，或者用户提供的路径可能有误
    # 这里假设路径正确，如果失败可能需要用户确认
    exit(1)

print(f"Original shape: {df.shape}")

# 2. 清洗
# 解析时间
print("Parsing datetime...")
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])

# 计算时长 (min)
df['duration_min'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60.0

# 过滤时长 [1, 180] min
original_len = len(df)
df = df[(df['duration_min'] >= 1) & (df['duration_min'] <= 180)]
print(f"Filtered by duration: {len(df)} (dropped {original_len - len(df)})")

# 计算 Haversine 距离 (km)
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

df['dist_km'] = haversine_np(
    df['start_location_x'], df['start_location_y'],
    df['end_location_x'], df['end_location_y']
)

# 计算速度 (km/h)
# 避免除以0，duration_min >= 1
df['speed_kmh'] = df['dist_km'] / (df['duration_min'] / 60.0)

# 过滤速度 <= 25 km/h
before_speed_filter = len(df)
df = df[df['speed_kmh'] <= 25]
print(f"Filtered by speed: {len(df)} (dropped {before_speed_filter - len(df)})")

# 坐标分位数过滤 [0.5%, 99.5%]
coords = ['start_location_x', 'start_location_y', 'end_location_x', 'end_location_y']
bounds = {}
for col in coords:
    lower = df[col].quantile(0.005)
    upper = df[col].quantile(0.995)
    bounds[col] = (lower, upper)
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print(f"Filtered by coordinates: {len(df)}")

# 去重
df = df.sort_values(by='duration_min', ascending=False).drop_duplicates(subset='orderid', keep='first')
print(f"Final shape after deduplication: {df.shape}")

# 保存清洗后数据
cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_data.csv")
df.to_csv(cleaned_path, index=False)
print(f"Saved cleaned data to {cleaned_path}")

# 3. 统计分析 (Q1部分)
# 全局小时分布
df['hour'] = df['start_time'].dt.hour
df['weekday'] = df['start_time'].dt.weekday  # 0=Monday, 6=Sunday
df['is_weekend'] = df['weekday'] >= 5

# 按小时聚合
hourly_counts = df.groupby(['is_weekend', 'hour']).size().reset_index(name='count')
# 为了取平均，需要除以该类别下的天数
# 统计数据集中包含的工作日和周末天数
unique_dates = df['start_time'].dt.date.unique()
weekend_dates = [d for d in unique_dates if d.weekday() >= 5]
weekday_dates = [d for d in unique_dates if d.weekday() < 5]
n_weekends = len(weekend_dates)
n_weekdays = len(weekday_dates)

print(f"Days: {len(unique_dates)}, Weekdays: {n_weekdays}, Weekends: {n_weekends}")

def normalize_count(row):
    if row['is_weekend']:
        return row['count'] / n_weekends if n_weekends > 0 else 0
    else:
        return row['count'] / n_weekdays if n_weekdays > 0 else 0

hourly_counts['avg_count'] = hourly_counts.apply(normalize_count, axis=1)

# 绘图
import matplotlib.font_manager as fm

# 尝试加载系统中的中文字体
font_candidates = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
]
for fpath in font_candidates:
    if os.path.exists(fpath):
        fm.fontManager.addfont(fpath)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 6))
sns.lineplot(data=hourly_counts, x='hour', y='avg_count', hue='is_weekend', marker='o')
plt.title('小时平均使用频率 (工作日 vs 周末)')
plt.xlabel('时刻')
plt.ylabel('平均使用次数')
plt.xticks(range(0, 24))
plt.grid(True)
plt.legend(title='是否周末')
plt.savefig(os.path.join(OUTPUT_DIR, "q1_hourly_usage.png"))
print("Saved q1_hourly_usage.png")

