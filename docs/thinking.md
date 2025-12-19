# 题目C：共享单车骑行分析与布局优化（问题1-3）建模与求解流程

> 数据来源：`题C-附件-mobike_shanghai_dataset.csv`（约 102,361 条记录）。  
> 时间范围：`start_time` ∈ [2016-08-01 00:23, 2016-08-31 23:58]；`end_time` 最晚到 2016-09-01 08:25。  
> 坐标系：WGS84（经度 `*_location_x`，纬度 `*_location_y`）。  
> 本文输出目标：给出可复现的（Q1 预测 → Q2 调度 → Q3 围栏）端到端流程。

---

## 0. 全流程概览

### 0.1 口径与参数表

| 模块 | 参数 | 取值 |
|---|---:|---|
| 时间离散 | $\Delta t$ | 1h |
| 时间起点 | $\tau_0$ | 2016-08-01 00:00 |
| 地球半径 | $R_\oplus$ | $6371000\text{ m}$ |
| 数值稳定项 | $\epsilon$ | $10^{-6}$ |
| 时长过滤 | $\Delta\tau$ | $[1,180]$ min |
| 速度过滤 | $v_{\max}$ | $25$ km/h |
| 坐标分位数过滤 | $(q_{\min},q_{\max})$ | $(0.5\%,99.5\%)$ |
| 站点聚类（Q1/Q2） | DBSCAN 半径 $r_{\text{st}}$ | $300\text{ m}$ |
| 站点聚类（Q1/Q2） | `min_samples` | $30$ |
| 主要站点 | Top-$K$ | $50$ |
| 预测窗口（服务） | $\mathcal{T}^{\text{pred}}$ | 次日 $07{:}00$–$09{:}00$（2h） |
| 调度窗口（运维） | $[T_s,T_e]$ | $00{:}00$–$06{:}00$（6h） |
| 需求安全系数 | $\rho$ | $15.0$ |
| 运维车容量 | $Q$ | $25$（辆） |
| 夜间平均速度 | $\nu$ | $25$ km/h |
| 车辆数安全系数 | $\eta$ | $1.3$ |
| 围栏候选聚类（Q3） | DBSCAN 半径 $r_f$ | $50\text{ m}$ |
| 围栏候选聚类（Q3） | `min_samples_f` | $5$ |
| 围栏最大推行距离 | $R$ | $200\text{ m}$ |
| 覆盖率阈值 | $\alpha$ | $1.0$ (100%) |
| 围栏候选规模 | Top-$M$ | All |

### 0.2 核心距离/时间计算（统一口径）

**（1）Haversine 距离（米）**  
对两点经纬度（弧度）$(\lambda_1,\varphi_1),(\lambda_2,\varphi_2)$：
$$
d = 2R_\oplus\arcsin\sqrt{\sin^2\frac{\varphi_2-\varphi_1}{2}+\cos\varphi_1\cos\varphi_2\sin^2\frac{\lambda_2-\lambda_1}{2}}
$$

**（2）平均速度（km/h）**  
$$
v_i=\frac{d_i/1000}{\Delta\tau_i/60}
$$

**（3）行驶时间（小时）**  
以站点中心点近似路网：$t_{ij}=\frac{d_{ij}/1000}{\nu}$（h）。

### 0.3 端到端流程（可复现）
1. **清洗**：时间解析 → 时长/速度过滤 → 坐标分位数过滤。  
2. **站点化（Q1/Q2）**：对所有起终点做 DBSCAN（$r_{\text{st}}=300$m, `min_samples`=30）→ 得到站点集合 $\mathcal{S}$ 与每单的 $s_i^o,s_i^d$。  
3. **小时聚合**：构造 $O_{s,t},I_{s,t}$（Outflow/Inflow）→ 计算 $F_s,TI_s$ → 选 Top-50 得 $\mathcal{S}^\star$。  
4. **预测（Q1）**：训练 ST-LSTM（站点+邻域）→ 预测次日 $07{:}00$–$09{:}00$ 的 $\widehat{O}_{s,t},\widehat{I}_{s,t}$。  
5. **再平衡需求（Q2 输入）**：由预测净流生成 $b_s$（投放为正、回收为负），并做“总量平衡修正”。  
6. **调度（Q2）**：构建 VRP-PD-TW（容量+时间窗）→ 改进 GA 近似求解车辆路径与装卸量。  
7.61→7. **围栏（Q3）**：以停车点密度生成候选围栏 → 覆盖选址（$\alpha=1.0,R=200$m）→ 贪心+剪枝输出围栏方案。

---

## 1. 问题一：使用频率分布、主要站点识别与需求预测

### 1.1 数据清洗
对每条订单 $i$：
1. **时间解析**：按 `%Y/%m/%d %H:%M` 解析 `start_time,end_time`。  
2. **时长过滤**：
$$
\Delta\tau_i\in[1,180]\text{ min}
$$
3. **坐标过滤（稳健分位数框）**：对所有起终点经度/纬度分别取 $[Q_{0.5\%},Q_{99.5\%}]$，超出者剔除。  
4. **速度过滤**：用起终点 Haversine 距离 $d_i$ 计算 $v_i$，保留 $v_i\le 25$ km/h。  
5. **去重**：以 `orderid` 为键去重（保留字段更完整/轨迹更长者）。

### 1.2 站点化（DBSCAN）
1. 构造点集（起终点并集）：
$$
\mathcal{P}=\{(x_i^s,y_i^s)\}_{i=1}^n\cup\{(x_i^e,y_i^e)\}_{i=1}^n
$$
2. DBSCAN（haversine）参数换算：$\varepsilon=r_{\text{st}}/R_\oplus$（弧度），取 $r_{\text{st}}=300$m；`min_samples`=30。  
3. 得到簇集合 $\mathcal{C}$，每簇中心作为站点 $s$ 坐标：
$$
(x_s,y_s)=\frac{1}{|\mathcal{P}_s|}\sum_{(x,y)\in\mathcal{P}_s}(x,y)
$$
4. 噪声点映射规则：若噪声点到最近站点距离 $\le 300$m，则吸附到最近站点；否则记为 $s=0$（“其他”）。

### 1.3 小时级指标与“潮汐效应”
令时间片索引：
$$
t=\left\lfloor\frac{\tau-\tau_0}{\Delta t}\right\rfloor,\quad \tau_0=\text{2016-08-01 00:00},\ \Delta t=1\text{h}
$$
对站点 $s$、时间片 $t$：
$$
O_{s,t}=\sum_i\mathbf{1}(s_i^{o}=s)\mathbf{1}(t_i^{o}=t),\qquad
I_{s,t}=\sum_i\mathbf{1}(s_i^{d}=s)\mathbf{1}(t_i^{d}=t)
$$
$$
U_{s,t}=O_{s,t}+I_{s,t},\qquad G_{s,t}=O_{s,t}-I_{s,t}
$$

**使用频率（站点强度）**：
$$
F_s=\frac{1}{|\mathcal{T}|}\sum_{t\in\mathcal{T}}U_{s,t},\qquad
P_s=\max_{t\in\mathcal{T}}U_{s,t}
$$

**潮汐指数**  
令早高峰集合 $\mathcal{T}_m=\{7,8\}$（小时-of-day），晚高峰 $\mathcal{T}_e=\{17,18\}$，其余为平峰 $\mathcal{T}_o$，则
$$
TI_s=\frac{\overline{U_{s,t}\mid t\in\mathcal{T}_m}+\overline{U_{s,t}\mid t\in\mathcal{T}_e}}{2\cdot \overline{U_{s,t}\mid t\in\mathcal{T}_o}+\epsilon}
$$

**全局时段分布（回答“不同时段使用频率”）**  
对所有站点汇总：
$$
O_t^{all}=\sum_{s\in\mathcal{S}}O_{s,t},\qquad I_t^{all}=\sum_{s\in\mathcal{S}}I_{s,t}
$$
令 $h(t)$ 为时间片 $t$ 对应的小时-of-day，则按小时-of-day 求均值曲线：
$$
\overline{O}(h)=\frac{1}{|\{t:h(t)=h\}|}\sum_{t:h(t)=h} O_t^{all},\qquad
\overline{I}(h)=\frac{1}{|\{t:h(t)=h\}|}\sum_{t:h(t)=h} I_t^{all}
$$
（可进一步按“工作日/周末”分组分别计算，用于展示潮汐差异。）

**OD 流向矩阵（主要站点）**  
对 $a,b\in\mathcal{S}^\star$ 定义 OD 频次：
$$
M_{a,b}=\sum_i \mathbf{1}(s_i^{o}=a)\mathbf{1}(s_i^{d}=b)
$$
取 $M_{a,b}$ 最大的若干 OD 对绘制箭线（线宽 ∝ $M_{a,b}$），用于解释早晚高峰的“潮汐迁移”。

### 1.4 主要站点（Top-50）
按 $F_s$ 从大到小排序，取 Top-$K=50$ 得主要站点集合 $\mathcal{S}^\star$。后续预测与调度仅在 $\mathcal{S}^\star$ 上进行（降维且符合“主要站点”题意）。

### 1.5 需求预测模型（ST-LSTM）

#### 1.5.1 输入输出定义
对每个站点 $s\in\mathcal{S}^\star$，定义二维序列：
$$
\mathbf{y}_{s,t}=[O_{s,t},I_{s,t}]^\top
$$

**外生时间特征**
$$
\mathbf{e}_t=\Big[\sin\frac{2\pi h}{24},\cos\frac{2\pi h}{24},\ \text{onehot}(\text{dow}),\ \mathbf{1}(\text{weekend})\Big]
$$
其中 $h$ 为小时-of-day，dow 为星期（0–6）。

#### 1.5.2 站点邻域与空间汇聚
对每个站点取最近 $k=10$ 个邻居 $\mathcal{N}(s)$，令 $\sigma$ 为“所有站点到其第 10 近邻距离”的中位数（m），定义权重：
$$
\alpha_{s,j}=\frac{\exp(-d_{s,j}/\sigma)}{\sum_{k\in\mathcal{N}(s)}\exp(-d_{s,k}/\sigma)},\qquad
\mathbf{m}_{s,t}=\sum_{j\in\mathcal{N}(s)}\alpha_{s,j}\mathbf{y}_{j,t}
$$
输入向量：
$$
\mathbf{x}_{s,t}=[\mathbf{y}_{s,t},\mathbf{m}_{s,t},\mathbf{e}_t]
$$

#### 1.5.3 ST-LSTM 结构与训练设置
- 序列长度：$L=48$（用过去 48h 预测未来）。  
- 预测步长：先做单步 $\widehat{\mathbf{y}}_{s,t+1}$，再滚动得到 $07{:}00$–$09{:}00$ 的 2 步预测。  
- 网络：2-layer LSTM，hidden=64，dropout=0.2；输出层线性 + ReLU 保证非负：
$$
\widehat{\mathbf{y}}_{s,t+1}=\mathrm{ReLU}(W\mathbf{h}_{s,t}+\mathbf{b})
$$
- 目标函数：MAE + L2 正则
$$
\min_\Theta\sum_{s,t}\left\|\widehat{\mathbf{y}}_{s,t+1}-\mathbf{y}_{s,t+1}\right\|_1+\lambda\|\Theta\|_2^2,\quad \lambda=10^{-4}
$$
- 数据变换：对 $O,I$ 做 $u=\log(1+u)$，并用训练集均值/方差标准化。  
- 训练：Adam，lr=$10^{-3}$，batch=32，epoch=50，early-stopping patience=5（以验证集 MAE）。

### 1.6 评估划分与输出接口

**时间切分**  
- Train：2016-08-01 至 2016-08-24  
- Val：2016-08-25 至 2016-08-27  
- Test：2016-08-28 至 2016-08-31（含跨日 inflow 到 2016-09-01 08:00）

**评价指标**
$$
\mathrm{MAE}=\frac{1}{n}\sum|y-\hat{y}|,\quad
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum(y-\hat{y})^2},\quad
\mathrm{SMAPE}=\frac{2}{n}\sum\frac{|y-\hat{y}|}{|y|+|\hat{y}|+\epsilon}
$$

**输出（交付物）**
1. 主要站点表：$s\in\mathcal{S}^\star$ 的 $(x_s,y_s),F_s,P_s,TI_s$。  
2. 小时使用频率分布：全局 $O_t^{all},I_t^{all}$ 的 hour-of-day 平均曲线（工作日/周末分开）。  
3. 次日预测：输出 $\widehat{O}_{s,t},\widehat{I}_{s,t}$（$s\in\mathcal{S}^\star$，$t$ 覆盖 07:00–09:00），作为 Q2 输入。  

【补充/可选/注意事项：1）若 DBSCAN 在 20 万点上运行较慢，可先将坐标投影到平面并用 BallTree/近邻索引加速，或先网格降采样再聚类；2）`track` 轨迹可用于更严格的异常检测（相邻点瞬时速度阈值）与通勤走廊识别，但不参与主流程；3）可增加 XGBoost 作为强基线并与 ST-LSTM 对比（论文更完整）；4）若要输出“热力图面状预测”，可在站点外再做规则网格并使用 ST-ResNet（作为可选扩展）。】

---

## 2. 问题二：夜间再平衡调度优化

### 2.1 再平衡需求构造（由 Q1 输出）
在预测窗口 $\mathcal{T}^{\text{pred}}$（次日 07:00–09:00）内定义净需求：
$$
g_s=\sum_{t\in\mathcal{T}^{\text{pred}}}\left(\widehat{O}_{s,t}-\widehat{I}_{s,t}\right)
$$
再平衡量（投放为正、回收为负）：
$$
b_s=\mathrm{round}(\rho\cdot g_s),\quad \rho=15.0
$$

**总量平衡修正**  
为满足“车辆守恒”（运维车从站点回收再投放，默认 Depot 不额外供车），令
$$
\Delta=\sum_{s\in\mathcal{S}^\star} b_s
$$
取 $s^\dagger=\arg\max_{s\in\mathcal{S}^\star}|b_s|$，修正：
$$
b_{s^\dagger}\leftarrow b_{s^\dagger}-\Delta
$$
使 $\sum_{s\in\mathcal{S}^\star}b_s=0$。

**车辆数 $m$**  
设总投放量 $B^+=\sum_{s\in\mathcal{S}^\star}\max(0,b_s)$，取
$$
m=\left\lceil \eta\cdot\frac{B^+}{Q}\right\rceil,\qquad \eta=1.3,\ Q=25
$$
（随后通过 $y_v$ 可自动少用车，但不超过 $m$）。

### 2.2 距离、时间与 Depot
1. 站点间距离 $d_{ij}$：用 Haversine（m）计算站点中心点距离。  
2. 成本：$c_{ij}=d_{ij}/1000$（km）。  
3. 行驶时间：$t_{ij}=(d_{ij}/1000)/\nu$（h），$\nu=25$ km/h。  
4. Depot 坐标：若无真实场站，取站点质心：
$$
(x_0,y_0)=\frac{1}{|\mathcal{S}^\star|}\sum_{s\in\mathcal{S}^\star}(x_s,y_s)
$$

### 2.3 数学规划模型（VRP-PD-TW）

#### 2.3.1 符号
- 节点集合：$\mathcal{N}=\{0,0'\}\cup \mathcal{S}^\star$（$0$ 出发 Depot，$0'$ 返回 Depot）。  
- 运维车辆：$\mathcal{V}=\{1,\dots,m\}$，容量 $Q=25$（辆）。  
- 时间窗：$[T_s,T_e]=[0,6]$（小时，表示 00:00–06:00）。  
- Big-M：$M_T=24$（h，用于时间窗线性化），$M_L=Q$（用于载重线性化）。  
- 服务时间：对站点 $s$，设
$$
s_s=\frac{2+0.2\cdot|b_s|}{60}\quad(\text{h})
$$
并取 Depot 服务时间 $s_0=s_{0'}=0$。

#### 2.3.2 变量
- $x_{vij}\in\{0,1\}$：车 $v$ 是否从 $i$ 到 $j$。  
- $y_v\in\{0,1\}$：车 $v$ 是否启用。  
- $y_{vs}\in\{0,1\}$：车 $v$ 是否服务站点 $s$。  
- $q_{vs}\in\mathbb{Z}$：车 $v$ 在站点 $s$ 的净装卸量（投放为正、回收为负）。  
- $L_{vi}\in[0,Q]$：车 $v$ 离开节点 $i$ 的载重（辆）。  
- $T_{vi}$：车 $v$ 到达节点 $i$ 的时间（h）。  
- 松弛 $e_s^+\ge 0,e_s^-\ge 0$：对 $b_s$ 的欠/超满足（罚函数，保证可解）。

#### 2.3.3 目标函数（权重）
设权重：
$$
w_1=1,\quad w_2=1000,\quad w_3=200,\quad w_4=50
$$
目标：
$$
\min Z=
w_1\sum_{v}\sum_{i}\sum_{j} c_{ij}x_{vij}
+ w_2\sum_{s} e_s^+
+ w_3\sum_{s} e_s^-
+ w_4\sum_{v} y_v
$$

#### 2.3.4 约束（核心）
1) 出入库：
$$
\sum_{j\in\mathcal{S}^\star}x_{v0j}=y_v,\qquad
\sum_{i\in\mathcal{S}^\star}x_{vi0'}=y_v,\qquad \forall v
$$
并禁止“回到出发 Depot / 从返回 Depot 再出发 / 自环”：
$$
\sum_{i\in\mathcal{N}}x_{vi0}=0,\quad
\sum_{j\in\mathcal{N}}x_{v0'j}=0,\quad
x_{v00'}=0,\quad
x_{vii}=0,\qquad \forall v,\forall i\in\mathcal{N}
$$

2) 访问与流守恒：
$$
\sum_{i\in\mathcal{N}}x_{vis}=\sum_{j\in\mathcal{N}}x_{vsj}=y_{vs},\qquad \forall v,\forall s\in\mathcal{S}^\star
$$
并与启用联动：
$$
y_{vs}\le y_v,\qquad \forall v,\forall s
$$
站点最多服务一次：
$$
\sum_{v}y_{vs}\le 1,\qquad \forall s
$$

3) 装卸量与访问联动：
$$
-Q\cdot y_{vs}\le q_{vs}\le Q\cdot y_{vs},\qquad \forall v,\forall s
$$

4) 需求满足（允许松弛）：
$$
\sum_v q_{vs}+e_s^+-e_s^-=b_s,\qquad \forall s
$$

5) 载重与容量（起始空车）：
$$
L_{v0}=0,\qquad 0\le L_{vi}\le Q,\qquad \forall v,\forall i\in\mathcal{N}
$$
若 $x_{vij}=1$ 且 $j$ 为站点，则
$$
L_{vj}=L_{vi}-q_{vj}\quad (\text{用 Big-M 线性化实现})
$$
即：
$$
L_{vj}\ge L_{vi}-q_{vj}-M_L(1-x_{vij}),\quad
L_{vj}\le L_{vi}-q_{vj}+M_L(1-x_{vij})
$$

6) 时间窗：
$$
T_{v0}=T_s=0,\qquad 0\le T_{vi}\le T_e=6,\qquad \forall v,\forall i\in\mathcal{N}
$$
若 $x_{vij}=1$：
$$
T_{vj}\ge T_{vi}+s_i+t_{ij}\quad (\text{Big-M 线性化})
$$
即：
$$
T_{vj}\ge T_{vi}+s_i+t_{ij}-M_T(1-x_{vij})
$$
并要求回库不晚于 $T_e$：
$$
T_{v0'}\le 6,\qquad \forall v
$$

7) 子回路消除（MTZ）：
对每车引入 $u_{vs}$（仅 $s\in\mathcal{S}^\star$）：
$$
u_{vi}-u_{vj}+1\le |\mathcal{S}^\star|\cdot(1-x_{vij}),\qquad \forall v,\forall i\neq j\in\mathcal{S}^\star
$$
$$
1\le u_{vs}\le |\mathcal{S}^\star|,\qquad \forall v,\forall s\in\mathcal{S}^\star
$$

### 2.4 求解算法（改进遗传算法 GA）

#### 2.4.1 编码与解码
1. **编码**：对 $\mathcal{S}^\star$（50 个站点）做一个排列（giant tour）。  
2. **Split 切分**：按排列顺序切分为 $m$ 条子路径（每条路径首尾加 Depot），目标最小化“距离 + 罚项”，并保证容量/时间窗尽量可行。  
3. **装卸量解码（贪心）**：  
   - 初始化每车载重 $L=0$，每站点剩余需求 $b_s^{rem}=b_s$；  
   - 访问站点 $s$ 时：  
     - 若 $b_s^{rem}<0$（需回收），回收量 $p=\min(-b_s^{rem},Q-L)$，令 $q=-p$，$L\leftarrow L+p$；  
     - 若 $b_s^{rem}>0$（需投放），投放量 $d=\min(b_s^{rem},L)$，令 $q=+d$，$L\leftarrow L-d$；  
     - 更新 $b_s^{rem}\leftarrow b_s^{rem}-q$。  
   - 路径结束时若仍有 $L>0$（车上剩余车），记为“未处置回收”，以罚项计入 $e^+$（或允许带回 Depot）。

#### 2.4.2 适应度函数
$$
\mathrm{fit}=
\text{Dist}
+ 1000\sum_s \max(0,|b_s^{rem}|)
+ 10^6\cdot \text{CapViol}
+ 10^6\cdot \text{TWViol}
+ 50\cdot (\#\text{used trucks})
$$

#### 2.4.3 GA 超参数
- 种群规模 $P=200$，代数 $G=500$；精英保留 10。  
- 交叉：OX（概率 0.9）；变异：Swap/Insert/Inverse（概率 0.2）。  
- 每代局部搜索：对每条路径做 2-opt（执行 1 轮）。  
- 终止：连续 50 代最优解不改进则提前停止。

### 2.5 输出与敏感性分析
**输出**：每辆车路径（站点序列）、各站点装卸量 $q_{vs}$、到达时间 $T_{vi}$、总里程/总成本。  
**指标**：
$$
\mathrm{SR}=1-\frac{\sum_s \max(0,e_s^+)}{\sum_s \max(0,b_s)+\epsilon}
$$

**敏感性分析**
- $Q\in\{20,25,30,40\}$；$\nu\in\{20,25,30\}$ km/h；$\rho\in\{1.0,1.1,1.2\}$；调度窗口长度 $\in\{5,6,7\}$ h。  
对每组参数运行 GA 10 次（不同随机种子），报告最优值与均值±标准差。

【补充/可选/注意事项：1）若允许车辆在 Depot 预装车（非空车出发），只需将 $L_{v0}$ 放松为 $[0,Q]$ 并在目标中加入“Depot 供给成本”；2）若希望更贴近现实，可放开“站点最多一次服务”，允许拆分装卸（多车/多次访问）；3）若需更强可解性，可用 OR-Tools VRP 或 MILP 求解器（Gurobi/CP-SAT）做小规模对比验证；4）若要动态调度，可将本模型作为 MPC 每小时滚动求解（超出本文范围）。】

---

## 3. 问题三：电子围栏布局优化

### 3.1 停放热点、候选围栏与需求权重

**（1）停车点集合**  
将每笔订单的起点与终点均视为一次停放事件：
$$
\mathcal{P}^{park}=\{(x_i^s,y_i^s)\}_{i=1}^n\cup\{(x_i^e,y_i^e)\}_{i=1}^n
$$

**（2）候选围栏点**  
对 $\mathcal{P}^{park}$ 运行 DBSCAN（haversine），半径 $r_f=50$m、`min_samples_f`=5（降低阈值以捕捉分散需求，确保用户便利），得到簇集合 $\mathcal{C}$；每簇中心作为候选围栏点 $j\in\mathcal{C}$，坐标 $(x_j,y_j)$。

**（3）需求点与权重**  
取需求点集合 $\mathcal{D}=\mathcal{C}$，权重定义为该簇内停放事件数量：
$$
w_i=\#\{p\in \mathcal{P}^{park}: p\ \text{属于簇 } i\}
$$

**（4）全覆盖策略**  
为了最大化用户便利性，我们不限制候选点数量，使用所有识别出的簇作为潜在围栏位置和需求点，目标是实现对所有需求点的 100% 覆盖。

### 3.2 覆盖约束选址模型（最小成本全覆盖）

最大推行距离 $R=200$ m，覆盖率阈值 $\alpha=1.0$（100%），单位成本 $f_j=1$。  
定义覆盖邻域：
$$
\mathcal{N}(i)=\{j\in\mathcal{C}: d_{ij}\le R\}
$$
决策变量：$y_j\in\{0,1\}$（是否建围栏），$z_i\in\{0,1\}$（需求点是否被覆盖）。

目标函数（最小化建设与维护成本）：
$$
\min \sum_{j\in\mathcal{C}} y_j
$$
约束（强制全覆盖）：
$$
z_i \le \sum_{j\in\mathcal{N}(i)} y_j,\qquad \forall i\in\mathcal{D}
$$
$$
\sum_{i\in\mathcal{D}} w_i z_i = \sum_{i\in\mathcal{D}} w_i \quad (\text{即 } z_i=1, \forall i)
$$

### 3.3 求解算法（贪心 + 剪枝）

**Step A：预计算覆盖集**  
对每个候选点 $j$ 计算 $\mathrm{Cov}(j)=\{i\in\mathcal{D}: d_{ij}\le R\}$。

**Step B：贪心加点 (Greedy Set Cover)**
1. 初始化 $S=\emptyset$，已覆盖集合 $U=\emptyset$。  
2. while $U \neq \mathcal{D}$（即未全覆盖）：选择
$$
j^\star=\arg\max_{j\notin S}\sum_{i\in \mathrm{Cov}(j)\setminus U} w_i
$$
更新 $S\leftarrow S\cup\{j^\star\}$，$U\leftarrow U\cup \mathrm{Cov}(j^\star)$。

**Step C：剪枝（Pruning，减少冗余围栏）**
按 $S$ 中围栏的“边际贡献”从小到大排序，依次尝试删除 $j\in S$：若删除后仍满足全覆盖约束（即覆盖集合 $U$ 仍等于 $\mathcal{D}$），则永久删除。重复直至无可删点。

### 3.4 输出与评价
**输出**：围栏集合 $S$ 的坐标列表、围栏数 $|S|$、覆盖率 $\mathrm{CR}$、未覆盖热点分布图。  
覆盖率：
$$
\mathrm{CR}=\frac{\sum_{i\in\mathcal{D}} w_i z_i}{\sum_{i\in\mathcal{D}} w_i}
$$

**敏感性分析**
- $R\in\{150,200,300\}$ m；$\alpha\in\{0.81,0.9,0.99\}$；$r_f\in\{45,50,55\}$ m；$M\in\{450,500,550\}$。  
对每组参数输出 $|S|$ 与 $\mathrm{CR}$。

**部分敏感性分析结果示例（固定 $r_f=50, M=500$）：**

| R (m) | Alpha | r_f (m) | M | 围栏数 \|S\| | 覆盖率 CR |
|---|---|---|---|---|---|
| 150 | 0.81 | 50 | 500 | 184 | 81.2% |
| 150 | 0.90 | 50 | 500 | 243 | 90.0% |
| 150 | 0.99 | 50 | 500 | 314 | 99.1% |
| 200 | 0.81 | 50 | 500 | 171 | 81.1% |
| 200 | 0.90 | 50 | 500 | 231 | 90.1% |
| 200 | 0.99 | 50 | 500 | 301 | 99.1% |
| 300 | 0.81 | 50 | 500 | 140 | 81.1% |
| 300 | 0.90 | 50 | 500 | 193 | 90.1% |
| 300 | 0.99 | 50 | 500 | 261 | 99.0% |

详细结果见 `data/output/q3_sensitivity_analysis.csv`。

**结果分析**
相比于允许部分覆盖的方案，全覆盖策略会显著增加围栏数量，但彻底解决了“用户无处还车”的痛点，实现了便利性最大化。通过剪枝算法，我们在满足全覆盖的前提下尽可能减少了冗余围栏，控制了成本。

【补充/可选/注意事项：1）若要体现“用户便利性”，可进一步计算平均/95分位推行距离（需引入分配变量或“就近围栏”规则）；2）若担心围栏超载，可加入围栏容量约束 $cap_j$；3）若希望与 Q1 更强耦合，可将权重 $w_i$ 改为“高峰期预测 Inflow”或“违规停放投诉权重”（需额外数据）；4）若候选点很多，可先网格化/H3 降维再做覆盖选址。】

---
