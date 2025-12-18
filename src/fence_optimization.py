import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from math import radians, sin, cos, sqrt, atan2
import os
import matplotlib.pyplot as plt
import logging

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
R_EARTH = 6371000  # m
R_FENCE = 200      # 覆盖半径 (m)
ALPHA = 1.0        # 覆盖率阈值 (100% full coverage)
EPS_METERS = 50    # DBSCAN 半径 (m)
MIN_SAMPLES = 5      # DBSCAN 最小样本 (Lowered to capture more demand)
TOP_M_CANDIDATES = 500
LOG_FILE = os.path.join(OUTPUT_DIR, "q3.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w'
)
logger = logging.getLogger()

def log_and_print(msg):
    print(msg)
    logger.info(msg)

# 1. 构造停放点集
log_and_print("Loading data for fence optimization...")
df = pd.read_csv(CLEANED_DATA_PATH)
# 停放点 = 起点 + 终点
points = pd.concat([
    df[['start_location_x', 'start_location_y']].rename(columns={'start_location_x': 'x', 'start_location_y': 'y'}),
    df[['end_location_x', 'end_location_y']].rename(columns={'end_location_x': 'x', 'end_location_y': 'y'})
], ignore_index=True)

log_and_print(f"Total parking events: {len(points)}")

# 2. 精细聚类 (候选点)
log_and_print("Running DBSCAN for candidates...")
coords = np.radians(points[['y', 'x']].values)
eps_rad = EPS_METERS / R_EARTH
db = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES, metric='haversine', algorithm='ball_tree', n_jobs=-1)
db.fit(coords)

points['cluster'] = db.labels_
n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
log_and_print(f"Found {n_clusters} clusters (candidate sites).")

# 3. 计算需求权重
# 每个簇是一个需求点，权重 = 簇内点数
# 候选围栏位置 = 簇中心
clusters = points[points['cluster'] != -1].groupby('cluster').agg({
    'x': 'mean',
    'y': 'mean',
    'cluster': 'count' # use cluster column to count rows
}).rename(columns={'cluster': 'weight'}).reset_index()

clusters.columns = ['cluster_id', 'x', 'y', 'weight']
total_weight = clusters['weight'].sum()
log_and_print(f"Total covered weight by clusters: {total_weight} ({total_weight/len(points):.2%} of total points)")

# 选取 Top M 候选点
if TOP_M_CANDIDATES is not None and len(clusters) > TOP_M_CANDIDATES:
    candidates = clusters.sort_values(by='weight', ascending=False).head(TOP_M_CANDIDATES).copy()
else:
    candidates = clusters.copy()
    
# 需求点集 = 候选点集 (简化：认为需求就集中在这些热点中心)
# 实际上需求点可以是所有原始点，但计算量太大。用簇中心代表需求点是合理的。
demand_points = candidates.copy() # index is cluster_id
candidate_points = candidates.copy() # index is cluster_id

n_demand = len(demand_points)
n_candidate = len(candidate_points)
print(f"Optimization problem size: {n_demand} demand points, {n_candidate} candidates")

# 4. 覆盖模型求解 (贪心)
# 计算距离矩阵 (需求点 i -> 候选点 j)
# 由于需求点和候选点集合相同，距离矩阵是对称的
# 但为了通用性，我们分开写
dists = np.zeros((n_demand, n_candidate))

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000 * c

# 向量化计算
# 这可能内存不足如果 M 很大，但 M=300 很小
dem_x = demand_points['x'].values
dem_y = demand_points['y'].values
cand_x = candidate_points['x'].values
cand_y = candidate_points['y'].values

# Expand dims for broadcasting
# dem: [N, 1], cand: [1, M]
lat1 = dem_y[:, np.newaxis]
lon1 = dem_x[:, np.newaxis]
lat2 = cand_y[np.newaxis, :]
lon2 = cand_x[np.newaxis, :]

# Haversine
dlat = np.radians(lat2 - lat1)
dlon = np.radians(lon2 - lon1)
lat1_rad = np.radians(lat1)
lat2_rad = np.radians(lat2)

a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
c = 2 * np.arcsin(np.sqrt(a))
dist_matrix = 6371000 * c # meters

# 覆盖关系矩阵
# cov_matrix[i, j] = 1 if candidate j covers demand i
cov_matrix = dist_matrix <= R_FENCE

# 贪心算法
selected_indices = []
covered_mask = np.zeros(n_demand, dtype=bool)
current_covered_weight = 0
target_weight = ALPHA * demand_points['weight'].sum()

log_and_print(f"Optimization problem size: {n_demand} demand points, {n_candidate} candidates")
log_and_print(f"Target covered weight: {target_weight}")

while current_covered_weight < target_weight:
    # 计算每个候选点的边际贡献 (能新覆盖多少权重)
    # 排除已选点 (虽然贪心通常不会重复选，但显式排除更好)
    best_candidate_idx = -1
    best_gain = -1
    
    # 剩余未覆盖的需求点索引
    uncovered_indices = np.where(~covered_mask)[0]
    
    if len(uncovered_indices) == 0:
        break
        
    # 只计算未覆盖点的贡献
    # gain[j] = sum(weight[i] for i in uncovered if covered_by[j])
    # 矩阵运算加速
    # valid_cov: [Uncovered_N, M]
    valid_cov = cov_matrix[uncovered_indices, :]
    # weights: [Uncovered_N]
    valid_weights = demand_points['weight'].values[uncovered_indices]
    
    # gain: [M]
    gains = np.dot(valid_weights, valid_cov)
    
    # 排除已选
    gains[selected_indices] = -1
    
    best_candidate_idx = np.argmax(gains)
    best_gain = gains[best_candidate_idx]
    
    if best_gain <= 0:
        print("No more gains possible.")
        break
        
    selected_indices.append(best_candidate_idx)
    
    # 更新覆盖状态
    new_covered = cov_matrix[:, best_candidate_idx]
    covered_mask = covered_mask | new_covered
    current_covered_weight = demand_points['weight'].values[covered_mask].sum()
    
    # print(f"Selected {len(selected_indices)}: Added {best_gain}, Total {current_covered_weight}/{target_weight}")

log_and_print(f"Greedy selection finished. Selected {len(selected_indices)} fences.")

# 剪枝 (Pruning)
# 尝试移除冗余点
# 按加入顺序逆序尝试？或者按边际贡献排序？
# 简单做法：遍历已选集合，如果移除某个点后覆盖率仍满足，则移除
# 为了更优，先尝试移除权重贡献最小的？
# 这里直接按顺序尝试
final_selected = selected_indices[:]
for idx in list(final_selected): # copy list to iterate
    # 尝试移除 idx
    temp_selected = [i for i in final_selected if i != idx]
    
    # 计算覆盖率
    if not temp_selected:
        continue
        
    # temp_cov_mask: [N]
    # any(cov_matrix[:, i] for i in temp_selected)
    # axis 1 sum > 0
    temp_cov_any = cov_matrix[:, temp_selected].any(axis=1)
    temp_weight = demand_points['weight'].values[temp_cov_any].sum()
    
    if temp_weight >= target_weight:
        final_selected.remove(idx)
        # print(f"Pruned candidate {idx}")

log_and_print(f"After pruning: {len(final_selected)} fences.")

# 5. 输出结果
selected_fences = candidate_points.iloc[final_selected].copy()
selected_fences['is_selected'] = True

# 未覆盖点
uncovered_mask = ~cov_matrix[:, final_selected].any(axis=1)
uncovered_points = demand_points[uncovered_mask].copy()

# 保存
selected_fences.to_csv(os.path.join(OUTPUT_DIR, "q3_fences.csv"), index=False)
log_and_print("Saved fences.")

# 计算最终覆盖率
final_cov_weight = demand_points['weight'].values[~uncovered_mask].sum()
final_ratio = final_cov_weight / demand_points['weight'].sum()
log_and_print(f"Final Coverage Ratio: {final_ratio:.4f}")

# 可视化
plt.figure(figsize=(10, 8))
# 绘制所有需求点 (灰色)
plt.scatter(demand_points['x'], demand_points['y'], c='lightgray', s=10, alpha=0.5, label='Demand Points')
# 绘制未覆盖点 (红色)
plt.scatter(uncovered_points['x'], uncovered_points['y'], c='red', s=10, alpha=0.8, label='Uncovered')
# 绘制围栏 (蓝色圈)
plt.scatter(selected_fences['x'], selected_fences['y'], c='blue', marker='x', s=50, label='Electronic Fences')

# 画围栏范围 (圆) - 稍微画几个示意的，画太多会乱
# for _, row in selected_fences.iterrows():
#     circle = plt.Circle((row['x'], row['y']), R_FENCE/100000, color='blue', fill=False, alpha=0.3) # 粗略比例
#     plt.gca().add_patch(circle)

plt.title(f'Electronic Fence Layout (Coverage: {final_ratio:.1%})')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "q3_fence_layout.png"))
log_and_print("Saved layout plot.")
