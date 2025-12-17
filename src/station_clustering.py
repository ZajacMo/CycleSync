import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from math import radians
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# 配置
CLEANED_DATA_PATH = "data/output/cleaned_data.csv"
OUTPUT_DIR = "data/output"
R_EARTH = 6371000  # 地球半径 (m)
EPS_METERS = 250   # 聚类半径 (m) - Larger radius to capture more points initially
MIN_SAMPLES = 30   # 最小样本数
SPLIT_THRESHOLD = 2000 # 如果一个 Cluster 超过这么多点，就拆分
TARGET_SIZE = 500      # 拆分后每个子 Cluster 的期望大小

print("Loading cleaned data...")
df = pd.read_csv(CLEANED_DATA_PATH)

# 1. 构造聚类点集 (Start & End)
# 为了统一编号，我们将起点和终点堆叠在一起进行聚类
points_start = df[['start_location_x', 'start_location_y']].copy()
points_start.columns = ['x', 'y']
points_start['type'] = 'start'
points_start['original_idx'] = df.index

points_end = df[['end_location_x', 'end_location_y']].copy()
points_end.columns = ['x', 'y']
points_end['type'] = 'end'
points_end['original_idx'] = df.index

all_points = pd.concat([points_start, points_end], ignore_index=True)
print(f"Total points for clustering: {len(all_points)}")

# 2. DBSCAN 聚类 (First Pass)
print("Running DBSCAN (First Pass)...")
# 将经纬度转换为弧度
coords = np.radians(all_points[['y', 'x']].values)
# eps 需要以弧度为单位: distance / R_EARTH
eps_rad = EPS_METERS / R_EARTH

db = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES, metric='haversine', algorithm='ball_tree', n_jobs=-1)
db.fit(coords)

all_points['cluster'] = db.labels_
n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print(f"DBSCAN found {n_clusters} clusters.")
print(f"Initial Noise points: {sum(db.labels_ == -1)}")

# 3. Refine Large Clusters with K-Means
print("Refining large clusters...")
unique_clusters = list(set(all_points['cluster']))
if -1 in unique_clusters:
    unique_clusters.remove(-1)

next_cluster_id = max(unique_clusters) + 1
final_labels = all_points['cluster'].copy()

for cid in unique_clusters:
    mask = all_points['cluster'] == cid
    cluster_points = all_points[mask]
    n_samples = len(cluster_points)
    
    if n_samples > SPLIT_THRESHOLD:
        n_sub_clusters = int(np.ceil(n_samples / TARGET_SIZE))
        print(f"  Splitting Cluster {cid} ({n_samples} points) into {n_sub_clusters} sub-clusters...")
        
        kmeans = KMeans(n_clusters=n_sub_clusters, random_state=42, n_init=10)
        sub_labels = kmeans.fit_predict(cluster_points[['x', 'y']])
        
        # Remap sub_labels to new unique IDs
        # To keep IDs distinct, we assign new IDs starting from next_cluster_id
        # The original ID (cid) will be replaced
        for i in range(n_sub_clusters):
            # We use (next_cluster_id + i) as the new ID
            # Update final_labels where cluster was cid AND sub_label is i
            # Need to map back to original indices
            indices = cluster_points.index[sub_labels == i]
            final_labels.loc[indices] = next_cluster_id + i
            
        next_cluster_id += n_sub_clusters
    else:
        # Keep as is, but maybe reassign ID to fill gaps if we wanted to be neat
        # For now, just keep original ID
        pass

all_points['cluster'] = final_labels
n_final_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
print(f"Final number of clusters after refinement: {n_final_clusters}")

# 3. 计算站点中心
cluster_centers = all_points[all_points['cluster'] != -1].groupby('cluster')[['x', 'y']].mean().reset_index()
cluster_centers.columns = ['cluster_id', 'center_x', 'center_y']
# 保存站点信息
cluster_centers.to_csv(os.path.join(OUTPUT_DIR, "station_centers.csv"), index=False)

# 4. 映射回原始数据
# 将聚类结果拆分回 start 和 end
# all_points 前一半是 start，后一半是 end
start_labels = all_points.iloc[:len(df)]['cluster'].values
end_labels = all_points.iloc[len(df):]['cluster'].values

df['start_cluster'] = start_labels
df['end_cluster'] = end_labels

# 噪声点处理：吸附到最近站点（<= 300m），否则归为 -1
# 这里为了简化，我们暂时保留 -1 作为“其他区域”
# 实际建模中，如果距离很近可以吸附，这里先略过吸附步骤，直接用聚类结果

# 5. 识别 Top-50 主要站点
# 统计每个站点的总流量 (In + Out)
# Outflow: start_cluster
out_counts = df[df['start_cluster'] != -1]['start_cluster'].value_counts()
# Inflow: end_cluster
in_counts = df[df['end_cluster'] != -1]['end_cluster'].value_counts()

total_flow = out_counts.add(in_counts, fill_value=0)
top_50_stations = total_flow.sort_values(ascending=False).head(50).index.tolist()

print(f"Top 50 stations identified: {top_50_stations}")
pd.Series(top_50_stations, name='station_id').to_csv(os.path.join(OUTPUT_DIR, "top_50_stations.csv"), index=False)

# 标记是否为主要站点
df['is_top_start'] = df['start_cluster'].isin(top_50_stations)
df['is_top_end'] = df['end_cluster'].isin(top_50_stations)

# 保存带聚类标签的数据
df.to_csv(os.path.join(OUTPUT_DIR, "clustered_data.csv"), index=False)
print("Saved clustered data.")

# 可视化：站点分布
plt.figure(figsize=(10, 8))
# 绘制所有站点
plt.scatter(cluster_centers['center_x'], cluster_centers['center_y'], s=10, c='blue', alpha=0.5, label='Stations')
# 绘制 Top 50
top_50_centers = cluster_centers[cluster_centers['cluster_id'].isin(top_50_stations)]
plt.scatter(top_50_centers['center_x'], top_50_centers['center_y'], s=50, c='red', marker='*', label='Top 50')
plt.title('Station Distribution and Top 50 Stations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "station_distribution.png"))
print("Saved station_distribution.png")
