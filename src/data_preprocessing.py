
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging

# 配置
DATA_PATH = "data/题C-附件-mobike_shanghai_dataset.csv"
OUTPUT_DIR = "data/output"
LOG_FILE = os.path.join(OUTPUT_DIR, "clean.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置日志
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w' # Overwrite each time
)
logger = logging.getLogger()

def log_and_print(msg):
    print(msg)
    logger.info(msg)

log_and_print("Starting data preprocessing...")

# 1. 读取数据
log_and_print("Loading data...")
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    log_and_print(f"Error reading CSV: {e}")
    exit(1)

initial_count = len(df)
log_and_print(f"Original record count: {initial_count}")

# 2. 清洗
# 解析时间
log_and_print("Parsing datetime...")
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])

# 计算时长 (min)
df['duration_min'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60.0

# 过滤时长 [1, 180] min
before_duration_filter = len(df)
df = df[(df['duration_min'] >= 1) & (df['duration_min'] <= 180)]
duration_dropped = before_duration_filter - len(df)
log_and_print(f"Filtered by duration (1-180 min): Dropped {duration_dropped} records. Remaining: {len(df)}")

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
speed_dropped = before_speed_filter - len(df)
log_and_print(f"Filtered by speed (<= 25 km/h): Dropped {speed_dropped} records. Remaining: {len(df)}")

# 坐标分位数过滤 [0.5%, 99.5%]
before_coord_filter = len(df)
coords = ['start_location_x', 'start_location_y', 'end_location_x', 'end_location_y']
bounds = {}
for col in coords:
    lower = df[col].quantile(0.005)
    upper = df[col].quantile(0.995)
    bounds[col] = (lower, upper)
    df = df[(df[col] >= lower) & (df[col] <= upper)]

coord_dropped = before_coord_filter - len(df)
log_and_print(f"Filtered by coordinates (0.5%-99.5% quantile): Dropped {coord_dropped} records. Remaining: {len(df)}")

# 去重
before_dedup = len(df)
df = df.sort_values(by='duration_min', ascending=False).drop_duplicates(subset='orderid', keep='first')
dedup_dropped = before_dedup - len(df)
log_and_print(f"Deduplication (by orderid): Dropped {dedup_dropped} records. Remaining: {len(df)}")

final_count = len(df)
total_dropped = initial_count - final_count
log_and_print(f"\nSummary:")
log_and_print(f"  Initial records: {initial_count}")
log_and_print(f"  Final records:   {final_count}")
log_and_print(f"  Total dropped:   {total_dropped} ({total_dropped/initial_count*100:.2f}%)")

# 保存清洗后数据
cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_data.csv")
df.to_csv(cleaned_path, index=False)
log_and_print(f"Saved cleaned data to {cleaned_path}")

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

log_and_print(f"Date range stats: Days={len(unique_dates)}, Weekdays={n_weekdays}, Weekends={n_weekends}")

hourly_counts['avg_count'] = 0.0
hourly_counts.loc[hourly_counts['is_weekend'], 'avg_count'] = hourly_counts.loc[hourly_counts['is_weekend'], 'count'] / n_weekends
hourly_counts.loc[~hourly_counts['is_weekend'], 'avg_count'] = hourly_counts.loc[~hourly_counts['is_weekend'], 'count'] / n_weekdays

# 绘图
plt.figure(figsize=(10, 6))
sns.lineplot(data=hourly_counts, x='hour', y='avg_count', hue='is_weekend', style='is_weekend', markers=True, dashes=False)
plt.title('Average Hourly Bike Usage: Weekday vs Weekend')
plt.xlabel('Hour of Day')
plt.ylabel('Average Number of Orders')
plt.xticks(range(0, 24))
plt.grid(True)
plt.legend(title='Is Weekend', labels=['Weekday', 'Weekend'])
plt.savefig(os.path.join(OUTPUT_DIR, "q1_hourly_usage.png"))
log_and_print(f"Saved q1_hourly_usage.png")
