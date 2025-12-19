# CycleSync: Spatio-Temporal Optimization for Urban Shared Mobility

共享单车骑行分析与布局优化 (Math Modeling)
---

本项目针对共享单车运营中的供需失衡、停车混乱等问题，建立了数据清洗、需求预测、调度优化和电子围栏布局的一整套数学模型解决方案。

## 项目结构

```
.
├── data/
│   └── output/             # 输出结果 (CSV, PNG)
├── src/
│   ├── data_preprocessing.py # 数据清洗与EDA
│   ├── station_clustering.py # 站点聚类 (DBSCAN)
│   ├── visualize_clusters.py # 聚类结果可视化
│   ├── demand_prediction.py  # 需求预测 (ST-LSTM)
│   ├── optimization.py       # 调度优化 (GA)
│   ├── fence_optimization.py # 电子围栏 (覆盖模型)
│   └── sensitivity_analysis_q3.py # 敏感性分析
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖包
├── 题目.md
├── thinking.md
└── 题C-附件-mobike_shanghai_dataset.csv
```

## 环境依赖

* Python 3.8+
* 依赖库：`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `tqdm`

安装依赖：
```bash
pip install -r requirements.txt
```

## 运行方式

直接运行主程序即可按顺序执行所有步骤：

```bash
python3 main.py
```

或者分别运行各个模块：

1. **数据清洗**: `python3 src/data_preprocessing.py`
2. **站点聚类**: `python3 src/station_clustering.py`
3. **需求预测**: `python3 src/demand_prediction.py`
4. **调度优化**: `python3 src/optimization.py`
5. **围栏布局**: `python3 src/fence_optimization.py`
6. **敏感性分析**: `python3 src/sensitivity_analysis_q3.py`

## 模型与结果说明

### 问题1：使用频率与需求预测

* **方法**: 
    * 使用 DBSCAN 对起终点进行聚类，识别出 81 个站点，并筛选出 Top-50 主要站点。
    * 构建 ST-LSTM 神经网络模型，利用过去 48 小时的流量特征和时间特征（小时、星期），预测次日 07:00-09:00 的借还需求。
* **结果**: 
    * 输出 `q1_predictions.csv`，包含各主要站点的预测出入流量。
    * 可视化图表：`q1_hourly_usage.png` (小时使用频率), `station_distribution.png` (站点分布), `q1_training_loss.png` (训练误差曲线)。

### 问题2：调度优化

* **方法**:
    * 基于预测的净流量计算各站点再平衡需求 ($b_s$)。
    * 建立 VRP-PD-TW (带时间窗和容量限制的车辆路径问题) 模型。
    * 使用遗传算法 (GA) 求解，目标是最小化运输成本和未满足需求惩罚。
* **结果**:
    * 输出了 11 辆车的调度路径方案 `q2_schedule.csv`。
    * 可视化路径图：`q2_routing.png`。

### 问题3：电子围栏布局

* **方法**:
    * 对所有停放点进行精细化 DBSCAN 聚类 ($r=50m$, `min_samples`=5) 得到候选围栏点。
    * 建立**最小成本全覆盖模型**，约束每个围栏覆盖半径 200m，目标覆盖率 **100%**。
    * 使用**贪心加点 (Greedy Set Cover)** 结合 **反向剪枝 (Pruning)** 策略求解，优先保证用户便利性（全覆盖），再最小化围栏数量。
* **结果**:
    * 选址方案 `q3_fences.csv`，共建设 **3165** 个电子围栏，实现了对所有 **10,684** 个细微需求点的 100% 覆盖。
    * 布局可视化：`q3_fence_layout.png`。
