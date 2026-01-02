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

# Default parameters
DEFAULT_R_FENCE = 200      # 覆盖半径 (m)
DEFAULT_ALPHA = 1.0        # 覆盖率阈值 (100% full coverage)
DEFAULT_EPS_METERS = 50    # DBSCAN 半径 (m)
DEFAULT_MIN_SAMPLES = 5    # DBSCAN 最小样本
DEFAULT_TOP_M_CANDIDATES = 500 # 候选点数量

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

def load_data(file_path):
    log_and_print("Loading data for fence optimization...")
    df = pd.read_csv(file_path)
    # 停放点 = 起点 + 终点
    points = pd.concat([
        df[['start_location_x', 'start_location_y']].rename(columns={'start_location_x': 'x', 'start_location_y': 'y'}),
        df[['end_location_x', 'end_location_y']].rename(columns={'end_location_x': 'x', 'end_location_y': 'y'})
    ], ignore_index=True)
    log_and_print(f"Total parking events: {len(points)}")
    return points

def generate_candidates(points, eps_meters, min_samples):
    log_and_print(f"Running DBSCAN with eps={eps_meters}m, min_samples={min_samples}...")
    coords = np.radians(points[['y', 'x']].values)
    eps_rad = eps_meters / R_EARTH
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine', algorithm='ball_tree', n_jobs=-1)
    db.fit(coords)
    
    points_labeled = points.copy()
    points_labeled['cluster'] = db.labels_
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    log_and_print(f"Found {n_clusters} clusters (candidate sites).")
    
    # Calculate cluster centers and weights
    clusters = points_labeled[points_labeled['cluster'] != -1].groupby('cluster').agg({
        'x': 'mean',
        'y': 'mean',
        'cluster': 'count'
    }).rename(columns={'cluster': 'weight'}).reset_index()
    
    clusters.columns = ['cluster_id', 'x', 'y', 'weight']
    return clusters

def optimize_layout(demand_points, candidate_points, r_fence, alpha):
    n_demand = len(demand_points)
    n_candidate = len(candidate_points)
    
    # dists
    dem_y = demand_points['y'].values[:, np.newaxis]
    dem_x = demand_points['x'].values[:, np.newaxis]
    cand_y = candidate_points['y'].values[np.newaxis, :]
    cand_x = candidate_points['x'].values[np.newaxis, :]
    
    dlat = np.radians(cand_y - dem_y)
    dlon = np.radians(cand_x - dem_x)
    lat1_rad = np.radians(dem_y)
    lat2_rad = np.radians(cand_y)
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    dist_matrix = R_EARTH * c 
    
    cov_matrix = dist_matrix <= r_fence
    
    selected_indices = []
    covered_mask = np.zeros(n_demand, dtype=bool)
    current_covered_weight = 0
    total_weight = demand_points['weight'].sum()
    target_weight = alpha * total_weight
    
    # Greedy
    while current_covered_weight < target_weight:
        uncovered_indices = np.where(~covered_mask)[0]
        if len(uncovered_indices) == 0:
            break
            
        valid_cov = cov_matrix[uncovered_indices, :]
        valid_weights = demand_points['weight'].values[uncovered_indices]
        gains = np.dot(valid_weights, valid_cov)
        
        gains[selected_indices] = -1
        
        best_candidate_idx = np.argmax(gains)
        best_gain = gains[best_candidate_idx]
        
        if best_gain <= 0:
            break
            
        selected_indices.append(best_candidate_idx)
        
        new_covered = cov_matrix[:, best_candidate_idx]
        covered_mask = covered_mask | new_covered
        current_covered_weight = demand_points['weight'].values[covered_mask].sum()
    
    # Pruning
    final_selected = selected_indices[:]
    for idx in list(final_selected):
        temp_selected = [i for i in final_selected if i != idx]
        if not temp_selected:
            continue
        
        temp_cov_any = cov_matrix[:, temp_selected].any(axis=1)
        temp_weight = demand_points['weight'].values[temp_cov_any].sum()
        
        if temp_weight >= target_weight:
            final_selected.remove(idx)
            
    # Result
    selected_fences = candidate_points.iloc[final_selected].copy()
    final_cov_weight = demand_points['weight'].values[cov_matrix[:, final_selected].any(axis=1)].sum()
    final_ratio = final_cov_weight / total_weight
    
    return selected_fences, final_ratio, cov_matrix[:, final_selected].any(axis=1)

def main():
    points = load_data(CLEANED_DATA_PATH)
    
    # 1. Generate Candidates (DBSCAN)
    clusters = generate_candidates(points, DEFAULT_EPS_METERS, DEFAULT_MIN_SAMPLES)
    
    # 2. Select Top M
    if DEFAULT_TOP_M_CANDIDATES is not None and len(clusters) > DEFAULT_TOP_M_CANDIDATES:
        candidates = clusters.sort_values(by='weight', ascending=False).head(DEFAULT_TOP_M_CANDIDATES).copy()
    else:
        candidates = clusters.copy()
        
    demand_points = candidates.copy()
    candidate_points = candidates.copy()
    
    log_and_print(f"Optimization problem size: {len(demand_points)} demand points, {len(candidate_points)} candidates")
    
    # 3. Optimize
    selected_fences, final_ratio, covered_mask = optimize_layout(
        demand_points, candidate_points, DEFAULT_R_FENCE, DEFAULT_ALPHA
    )
    
    log_and_print(f"Selected {len(selected_fences)} fences.")
    log_and_print(f"Final Coverage Ratio: {final_ratio:.4f}")
    
    # 4. Save and Plot
    selected_fences.to_csv(os.path.join(OUTPUT_DIR, "q3_fences.csv"), index=False)
    
    uncovered_points = demand_points[~covered_mask].copy()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(demand_points['x'], demand_points['y'], c='lightgray', s=10, alpha=0.5, label='需求点')
    plt.scatter(uncovered_points['x'], uncovered_points['y'], c='red', s=10, alpha=0.8, label='未覆盖')
    plt.scatter(selected_fences['x'], selected_fences['y'], c='blue', marker='x', s=50, label='电子围栏')
    plt.title(f'电子围栏布局（覆盖率：{final_ratio:.1%}）')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "q3_fence_layout.png"))
    log_and_print("Saved results.")

if __name__ == "__main__":
    main()
