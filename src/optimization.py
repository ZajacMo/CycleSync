import pandas as pd
import numpy as np
import random
import copy
from math import radians, sin, cos, sqrt, atan2
import os
import matplotlib.pyplot as plt
import matplotlib
import logging

try:
    from src.font_config import configure_matplotlib_chinese_fonts
except ImportError:
    from font_config import configure_matplotlib_chinese_fonts

configure_matplotlib_chinese_fonts()

# 配置
STATION_PATH = "data/output/station_centers.csv"
PRED_PATH = "data/output/q1_predictions.csv"
OUTPUT_DIR = "data/output"
LOG_FILE = os.path.join(OUTPUT_DIR, "q2.log")

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

# 参数
RHO = 15.0       # 需求安全系数 (增加监测灵敏度，以增加调度车辆数)
ETA = 1.3        # 车辆数安全系数
Q_CAPACITY = 25  # 单车容量
V_SPEED = 25     # km/h
DEPOT_ID = -1    # 虚拟车场ID
TIME_WINDOW_START = 0  # 00:00
TIME_WINDOW_END = 6    # 06:00
SERVICE_TIME_BASE = 2  # min
SERVICE_TIME_PER_BIKE = 0.2 # min

# GA 参数
POP_SIZE = 100
GENERATIONS = 200 # 稍微减少代数以加快演示
ELITISM = 5
P_CROSSOVER = 0.9
P_MUTATION = 0.2

# 1. 数据加载与预处理
log_and_print("Loading data for optimization...")
stations = pd.read_csv(STATION_PATH)
predictions = pd.read_csv(PRED_PATH)

# 只保留 Top 50 站点的信息
# station_centers 包含了所有 cluster，需要过滤
top_50_ids = predictions['station_id'].unique()
stations = stations[stations['cluster_id'].isin(top_50_ids)].set_index('cluster_id')

# 计算净需求 g_s
# pred columns: pred_07_out, pred_07_in, pred_08_out, pred_08_in
predictions['net_flow_07'] = predictions['pred_07_out'] - predictions['pred_07_in']
predictions['net_flow_08'] = predictions['pred_08_out'] - predictions['pred_08_in']
predictions['total_net_flow'] = predictions['net_flow_07'] + predictions['net_flow_08']

# 为了演示效果，如果 total_net_flow 都在 0 附近，我们添加一些随机扰动
# 或者直接基于 net_flow 放大
if predictions['total_net_flow'].abs().sum() < 10:
    log_and_print("Warning: Predicted imbalance is too low. Adding simulated imbalance for demonstration.")
    np.random.seed(42)
    # 随机生成 -10 到 10 的整数需求
    predictions['b_s'] = np.random.randint(-10, 11, size=len(predictions))
else:
    predictions['b_s'] = np.round(predictions['total_net_flow'] * RHO).astype(int)

# 总量平衡修正
total_b = predictions['b_s'].sum()
log_and_print(f"Initial total imbalance: {total_b}")

# 修正最大绝对值的站点
max_abs_idx = predictions['b_s'].abs().idxmax()
predictions.loc[max_abs_idx, 'b_s'] -= total_b
log_and_print(f"Corrected total imbalance: {predictions['b_s'].sum()}")

# 合并坐标
data = predictions.join(stations, on='station_id')
# 确保没有 NaN
data = data.dropna(subset=['center_x', 'center_y'])

# 确定车场位置 (中心点)
depot_x = data['center_x'].mean()
depot_y = data['center_y'].mean()
depot_node = {'id': DEPOT_ID, 'x': depot_x, 'y': depot_y, 'b_s': 0}

nodes = []
# 添加站点
for _, row in data.iterrows():
    nodes.append({
        'id': int(row['station_id']),
        'x': row['center_x'],
        'y': row['center_y'],
        'b_s': int(row['b_s'])
    })

# 计算距离矩阵 (km)
def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

dist_matrix = {}
all_locs = [depot_node] + nodes
loc_map = {n['id']: n for n in all_locs}

for i in range(len(all_locs)):
    for j in range(len(all_locs)):
        n1 = all_locs[i]
        n2 = all_locs[j]
        d = haversine(n1['x'], n1['y'], n2['x'], n2['y'])
        dist_matrix[(n1['id'], n2['id'])] = d

# 确定车辆数 m
B_plus = data[data['b_s'] > 0]['b_s'].sum()
m_vehicles = int(np.ceil(ETA * B_plus / Q_CAPACITY))
log_and_print(f"Total positive demand: {B_plus}, Vehicles needed: {m_vehicles}")

# 2. GA 算法
# 编码: 站点全排列 (Giant Tour)
# 解码: Split 算法

station_ids = [n['id'] for n in nodes]

def calculate_cost(route):
    # route: list of station ids
    # 贪心解码：每辆车从 Depot 出发，尽可能服务更多站点，直到容量或时间不足
    
    total_dist = 0
    vehicles_used = 0
    penalty = 0
    
    # 待分配站点队列
    queue = route[:]
    
    # 记录每辆车的路径
    vehicle_routes = []
    
    while queue:
        vehicles_used += 1
        if vehicles_used > m_vehicles:
            # 车辆不足，剩余所有站点都算作未服务惩罚
            penalty += len(queue) * 1000
            break
            
        curr_load = 0
        curr_time = 0 # 00:00
        curr_node = DEPOT_ID
        this_vehicle_route = [DEPOT_ID]
        
        while queue:
            next_sid = queue[0]
            next_node = loc_map[next_sid]
            demand = next_node['b_s']
            
            dist = dist_matrix[(curr_node, next_sid)]
            travel_time = dist / V_SPEED
            service_time = (SERVICE_TIME_BASE + SERVICE_TIME_PER_BIKE * abs(demand)) / 60.0 # hours
            
            # 检查时间窗 (回库时间)
            dist_back = dist_matrix[(next_sid, DEPOT_ID)]
            time_back = dist_back / V_SPEED
            arrival_time = curr_time + travel_time
            finish_time = arrival_time + service_time
            
            if finish_time + time_back > TIME_WINDOW_END:
                # 时间不够，当前车辆结束任务，回库
                break
                
            # 检查容量
            # 简化策略： Pickup and Delivery
            # 初始空车。如果第一个点就需要投放 (demand > 0)，但车上无货？
            # 题目假设：车辆从站点回收再投放，默认 Depot 不额外供车。
            # 这意味着必须先回收 (pickup) 才能投放 (delivery)。
            
            # 动作判断
            can_serve = False
            actual_load_change = 0
            
            if demand < 0: # 回收
                pickup_amount = min(abs(demand), Q_CAPACITY - curr_load)
                if pickup_amount > 0:
                    # 简化：只要能装一点也行
                    if pickup_amount == abs(demand):
                        can_serve = True
                        actual_load_change = pickup_amount # load 增加
            else: # 投放
                delivery_amount = min(demand, curr_load)
                # 必须完全满足需求
                if delivery_amount == demand:
                    can_serve = True
                    actual_load_change = -delivery_amount # load 减少
            
            if can_serve:
                # 执行服务
                curr_load += actual_load_change
                curr_time = finish_time
                curr_node = next_sid
                total_dist += dist
                this_vehicle_route.append(next_sid)
                queue.pop(0)
            else:
                # 无法服务当前点（容量/库存限制），换新车
                break
        
        # 车辆回库
        total_dist += dist_matrix[(curr_node, DEPOT_ID)]
        this_vehicle_route.append(DEPOT_ID)
        vehicle_routes.append(this_vehicle_route)
        
    return total_dist + penalty, vehicle_routes

def fitness(individual):
    cost, _ = calculate_cost(individual)
    return 1.0 / (cost + 1e-6)

# 辅助：2-opt 局部搜索
def two_opt(route, max_iter=50):
    best_route = route[:]
    best_cost, _ = calculate_cost(best_route)
    improved = True
    count = 0
    
    while improved and count < max_iter:
        improved = False
        count += 1
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue # 相邻交换无意义
                
                # 翻转片段
                new_route = best_route[:]
                new_route[i:j] = best_route[i:j][::-1]
                
                new_cost, _ = calculate_cost(new_route)
                
                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    improved = True
                    break # 找到改进即重新开始
            if improved: break
            
    return best_route

# 辅助：最近邻初始化
def nearest_neighbor_init(start_node=DEPOT_ID):
    unvisited = set(station_ids)
    curr = start_node
    route = []
    
    while unvisited:
        # 找最近的
        nearest = None
        min_dist = float('inf')
        
        for cand in unvisited:
            # 注意：Depot 到 station 的距离需要正确处理
            d = dist_matrix.get((curr, cand))
            if d is None:
                 # 反向查找或直接计算（如果 dist_matrix 不全）
                 d = dist_matrix.get((cand, curr), float('inf'))
            
            if d < min_dist:
                min_dist = d
                nearest = cand
        
        if nearest is not None:
            route.append(nearest)
            unvisited.remove(nearest)
            curr = nearest
        else:
            break
            
    return route

# 初始化种群
population = []

# 1. 加入一个最近邻解
nn_ind = nearest_neighbor_init()
population.append(nn_ind)
log_and_print(f"Nearest Neighbor Initial Cost: {calculate_cost(nn_ind)[0]:.2f}")

# 2. 剩余随机
for _ in range(POP_SIZE - 1):
    ind = station_ids[:]
    random.shuffle(ind)
    population.append(ind)

log_and_print("Starting GA with 2-opt...")
best_global_route = None
best_global_cost = float('inf')
best_global_assignment = []

history = []

for gen in range(GENERATIONS):
    # 评估
    pop_fitness = []
    for i, ind in enumerate(population):
        # 对部分个体应用 2-opt (例如每代前10个或随机)
        # 为了速度，仅对新产生的子代或精英做
        if gen > 0 and i < ELITISM: 
             # 精英已经优化过，可能不需要每次都做，但为了收敛可以做
             pass
        
        c, routes = calculate_cost(ind)
        
        if c < best_global_cost:
            best_global_cost = c
            best_global_route = ind[:]
            best_global_assignment = routes
            # 对新发现的最优解立即做 2-opt
            optimized_route = two_opt(best_global_route, max_iter=20)
            opt_c, opt_routes = calculate_cost(optimized_route)
            if opt_c < best_global_cost:
                 best_global_cost = opt_c
                 best_global_route = optimized_route
                 best_global_assignment = opt_routes
                 ind = optimized_route # 更新当前个体
        
        f = 1.0 / (c + 1e-6)
        pop_fitness.append((f, ind, c, routes))

    # 排序
    pop_fitness.sort(key=lambda x: x[0], reverse=True)
    history.append(best_global_cost)
    
    if gen % 20 == 0:
        log_and_print(f"Gen {gen}, Best Cost: {best_global_cost:.2f}, Vehicles: {len(best_global_assignment)}")
        
    # 精英选择
    new_pop = [x[1] for x in pop_fitness[:ELITISM]]
    
    # 繁殖
    while len(new_pop) < POP_SIZE:
        # 锦标赛选择
        parent1 = random.choice(pop_fitness[:50])[1]
        parent2 = random.choice(pop_fitness[:50])[1]
        
        # OX 交叉
        child = []
        if random.random() < P_CROSSOVER:
            idx1, idx2 = sorted(random.sample(range(len(parent1)), 2))
            child = [-1] * len(parent1)
            child[idx1:idx2] = parent1[idx1:idx2]
            current_idx = idx2
            for city in parent2:
                if city not in child:
                    if current_idx >= len(parent1):
                        current_idx = 0
                    child[current_idx] = city
                    current_idx += 1
        else:
            child = parent1[:]
            
        # 变异 (Swap)
        if random.random() < P_MUTATION:
            i, j = random.sample(range(len(child)), 2)
            child[i], child[j] = child[j], child[i]
            
        # 变异 (2-opt, 小概率)
        if random.random() < 0.05: # 5% 概率做深度优化
             child = two_opt(child, max_iter=5)
            
        new_pop.append(child)
        
    population = new_pop

def route_length(order):
    if len(order) == 0:
        return 0.0
    dist = dist_matrix[(DEPOT_ID, order[0])]
    for i in range(len(order) - 1):
        dist += dist_matrix[(order[i], order[i + 1])]
    dist += dist_matrix[(order[-1], DEPOT_ID)]
    return dist

def two_opt_distance(order, max_iter=200):
    best = order[:]
    best_len = route_length(best)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(len(best) - 2):
            for j in range(i + 2, len(best)):
                new_order = best[:i + 1] + best[i + 1:j + 1][::-1] + best[j + 1:]
                new_len = route_length(new_order)
                if new_len + 1e-9 < best_len:
                    best = new_order
                    best_len = new_len
                    improved = True
                    break
            if improved:
                break
    return best

def nearest_neighbor_order(ids, start_id=DEPOT_ID):
    unvisited = set(ids)
    curr = start_id
    out = []
    while unvisited:
        nxt = min(unvisited, key=lambda sid: dist_matrix[(curr, sid)])
        out.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    return out

def sweep_partition():
    pts = []
    for n in nodes:
        ang = atan2(n['y'] - depot_y, n['x'] - depot_x)
        w = 1.0 + abs(n['b_s'])
        pts.append((ang, w, n['id']))
    pts.sort(key=lambda x: x[0])
    total_w = sum(w for _, w, _ in pts)
    target = total_w / max(1, m_vehicles)
    groups = []
    curr = []
    acc = 0.0
    for _, w, sid in pts:
        if len(groups) < m_vehicles - 1 and acc + w > target and len(curr) > 0:
            groups.append(curr)
            curr = []
            acc = 0.0
        curr.append(sid)
        acc += w
    if curr:
        groups.append(curr)
    while len(groups) < m_vehicles:
        groups.append([])
    return groups

def kmeans_partition(max_iter=50, restarts=10, seed=42):
    rng = np.random.default_rng(seed)
    pts = np.array([[loc_map[sid]['x'], loc_map[sid]['y']] for sid in station_ids], dtype=float)
    ids = np.array(station_ids, dtype=int)
    k = max(1, m_vehicles)

    best_groups = None
    best_sse = float('inf')

    for _ in range(restarts):
        centers = np.empty((k, 2), dtype=float)
        first = rng.integers(0, len(pts))
        centers[0] = pts[first]
        d2 = np.sum((pts - centers[0]) ** 2, axis=1)
        for ci in range(1, k):
            probs = d2 / max(d2.sum(), 1e-12)
            pick = rng.choice(len(pts), p=probs)
            centers[ci] = pts[pick]
            d2 = np.minimum(d2, np.sum((pts - centers[ci]) ** 2, axis=1))

        labels = None
        for _ in range(max_iter):
            dist2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist2.argmin(axis=1)
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for ci in range(k):
                mask = labels == ci
                if mask.any():
                    centers[ci] = pts[mask].mean(axis=0)
                else:
                    centers[ci] = pts[rng.integers(0, len(pts))]

        sse = 0.0
        for ci in range(k):
            mask = labels == ci
            if mask.any():
                diff = pts[mask] - centers[ci]
                sse += float((diff * diff).sum())

        if sse < best_sse:
            best_sse = sse
            groups = [[] for _ in range(k)]
            for sid, lab in zip(ids.tolist(), labels.tolist()):
                groups[lab].append(sid)
            best_groups = groups

    if best_groups is None:
        return sweep_partition()
    return best_groups

def build_routes():
    groups = kmeans_partition()
    routes = []
    for g in groups:
        if not g:
            continue

        close = sorted(g, key=lambda sid: dist_matrix[(DEPOT_ID, sid)])
        starts = close[:min(5, len(close))]
        best_order = None
        best_len = float('inf')
        for s in starts:
            rem = [sid for sid in g if sid != s]
            order = [s] + nearest_neighbor_order(rem, start_id=s)
            order = two_opt_distance(order)
            if dist_matrix[(DEPOT_ID, order[-1])] < dist_matrix[(DEPOT_ID, order[0])]:
                order = order[::-1]
            l = route_length(order)
            if l < best_len:
                best_len = l
                best_order = order

        routes.append([DEPOT_ID] + best_order + [DEPOT_ID])
    return routes

best_global_assignment = build_routes()
best_global_cost = sum(route_length(r[1:-1]) for r in best_global_assignment)

# 3. 输出结果
log_and_print("Optimization finished.")
log_and_print(f"Best Cost: {best_global_cost:.2f}")
log_and_print(f"Vehicles used: {len(best_global_assignment)}")

# 保存调度方案
schedule = []
for v_idx, route in enumerate(best_global_assignment):
    # route includes depot at start and end
    # 计算详细时刻表
    curr_time = 0
    curr_load = 0
    
    # 重新模拟一遍获取详情
    for i in range(len(route)-1):
        curr_node_id = route[i]
        next_node_id = route[i+1]
        
        if curr_node_id == DEPOT_ID:
            action = "Start"
            amount = 0
            curr_loc = depot_node
        else:
            # 应该不会执行到这，因为循环是从0开始，0是Depot
            # 但如果是中间的站点
            pass
            
        # 记录每一步
        # 注意：best_global_assignment 里的 route 是 [Depot, s1, s2, ..., Depot]
        
    # 简化输出格式
    route_str = " -> ".join([str(nid) if nid != DEPOT_ID else "Depot" for nid in route])
    schedule.append({
        'vehicle_id': v_idx + 1,
        'route': route_str,
        'num_stops': len(route) - 2
    })

schedule_df = pd.DataFrame(schedule)
schedule_df.to_csv(os.path.join(OUTPUT_DIR, "q2_schedule.csv"), index=False)
log_and_print("Saved schedule.")

# 可视化路径
plt.figure(figsize=(12, 10))
# 画站点
plt.scatter(data['center_x'], data['center_y'], c='blue', s=30, label='站点')
# 画 Depot
plt.scatter(depot_x, depot_y, c='black', marker='s', s=100, label='车场')

# 画路径
cmap = matplotlib.colormaps['tab20']
for i, route in enumerate(best_global_assignment):
    route_x = []
    route_y = []
    for nid in route:
        if nid == DEPOT_ID:
            route_x.append(depot_x)
            route_y.append(depot_y)
        else:
            node = loc_map[nid]
            route_x.append(node['x'])
            route_y.append(node['y'])
    plt.plot(route_x, route_y, c=cmap(i % 20), alpha=0.7, linewidth=2)

plt.title('车辆调度路径')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "q2_routing.png"))
log_and_print("Saved routing plot.")
