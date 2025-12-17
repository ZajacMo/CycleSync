import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

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
CLUSTERED_DATA_PATH = "data/output/clustered_data.csv"
TOP_50_PATH = "data/output/top_50_stations.csv"
OUTPUT_DIR = "data/output"
PRED_WINDOW_START = pd.Timestamp("2016-09-01 07:00:00")
PRED_WINDOW_END = pd.Timestamp("2016-09-01 09:00:00")

# 模型参数
SEQ_LEN = 48  # 过去48小时
PRED_LEN = 1  # 预测未来1小时（滚动预测）
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# 1. 数据准备
print("Preparing data...")
df = pd.read_csv(CLUSTERED_DATA_PATH)
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
top_50 = pd.read_csv(TOP_50_PATH)['station_id'].tolist()

# 过滤只保留 Top 50 相关记录 (用于训练)
# 注意：这里我们只关心 Top 50 站点的 In/Out，
# 对于非 Top 50 的流动，我们暂不作为主要特征，或者归为“其他”
# 为了构建完整的时序矩阵，我们需要对 Top 50 每个站点生成每小时的 In/Out

# 生成完整的时间索引 (从数据最早时间到预测结束时间的前一个小时)
# 数据范围: 2016-08-01 到 2016-08-31
# 预测目标: 2016-09-01 07:00, 08:00
# 因此我们需要构造时间轴直到 2016-09-01 08:00
time_idx = pd.date_range(start="2016-08-01 00:00:00", end="2016-09-01 08:00:00", freq='h')
n_timesteps = len(time_idx)
n_stations = len(top_50)

print(f"Time steps: {n_timesteps}, Stations: {n_stations}")

# 初始化流量矩阵: [Time, Station, 2] (0: Out, 1: In)
flow_matrix = np.zeros((n_timesteps, n_stations, 2))

# 映射 station_id 到 0..49
station_map = {sid: i for i, sid in enumerate(top_50)}

# 填充 Outflow
# 按照 start_time 聚合
df['start_hour'] = df['start_time'].dt.floor('h')
out_counts = df[df['start_cluster'].isin(top_50)].groupby(['start_hour', 'start_cluster']).size()

for (ts, sid), count in out_counts.items():
    if ts in time_idx:
        t_idx = time_idx.get_loc(ts)
        s_idx = station_map[sid]
        flow_matrix[t_idx, s_idx, 0] = count

# 填充 Inflow
# 按照 end_time 聚合
df['end_hour'] = df['end_time'].dt.floor('h')
in_counts = df[df['end_cluster'].isin(top_50)].groupby(['end_hour', 'end_cluster']).size()

for (ts, sid), count in in_counts.items():
    if ts in time_idx:
        t_idx = time_idx.get_loc(ts)
        s_idx = station_map[sid]
        flow_matrix[t_idx, s_idx, 1] = count

# 2. 特征工程
# Log transform
flow_matrix = np.log1p(flow_matrix)

# Standardization
scaler = StandardScaler()
# Reshape to [Time * Station, 2] for scaling
flow_flat = flow_matrix.reshape(-1, 2)
scaler.fit(flow_flat)
flow_scaled = scaler.transform(flow_flat).reshape(n_timesteps, n_stations, 2)

# 时间特征 (Time of Day, Day of Week)
# shape: [Time, 2] -> sin/cos hour, one-hot dow (simplified to just is_weekend for now based on thinking.md plan, 
# but let's use sin/cos for hour and simple normalization for day)
hour_feature = np.sin(2 * np.pi * time_idx.hour / 24.0)
day_feature = time_idx.dayofweek / 6.0  # Normalize 0-6 to 0-1
time_features = np.stack([hour_feature, day_feature], axis=1) # [Time, 2]

# 3. Dataset
class FlowDataset(Dataset):
    def __init__(self, flow_data, time_data, seq_len=48, pred_len=1):
        self.flow_data = torch.FloatTensor(flow_data)
        self.time_data = torch.FloatTensor(time_data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.flow_data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        # Input: [Seq, Station, 2]
        # Time feat: [Seq, 2] (broadcast to stations later or concat)
        
        # 为了简化 ST-LSTM，我们将所有站点的流量展平或者作为 Batch 的一部分
        # 这里我们将 Station 维度保留，模型中处理
        # Input X: Flow[t:t+L], Time[t:t+L+1] (future time known)
        
        x_flow = self.flow_data[idx : idx+self.seq_len] # [L, S, 2]
        x_time = self.time_data[idx : idx+self.seq_len] # [L, 2]
        
        y_flow = self.flow_data[idx+self.seq_len : idx+self.seq_len+self.pred_len] # [P, S, 2]
        y_time = self.time_data[idx+self.seq_len : idx+self.seq_len+self.pred_len] # [P, 2] # Future time features
        
        return (x_flow, x_time), (y_flow, y_time)

# 划分 Train/Val/Test
# Train: 8.1 - 8.24
# Val: 8.25 - 8.27
# Test: 8.28 - 8.31 (留出最后一天用于生成 9.1 的预测)
# 注意：我们要预测 9.1，所以训练数据用到 8.31 是合理的。
# 这里的 Test 集主要用于评估模型性能。
# 实际预测 9.1 时，我们将使用 8.30-8.31 的数据作为输入。

split_train = pd.Timestamp("2016-08-25 00:00:00")
split_val = pd.Timestamp("2016-08-28 00:00:00")

train_idx = time_idx.get_loc(split_train)
val_idx = time_idx.get_loc(split_val)

train_data = FlowDataset(flow_scaled[:train_idx], time_features[:train_idx], SEQ_LEN, PRED_LEN)
val_data = FlowDataset(flow_scaled[train_idx-SEQ_LEN:val_idx], time_features[train_idx-SEQ_LEN:val_idx], SEQ_LEN, PRED_LEN)
# Test set for evaluation
test_data = FlowDataset(flow_scaled[val_idx-SEQ_LEN:], time_features[val_idx-SEQ_LEN:], SEQ_LEN, PRED_LEN)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# 4. 模型定义
class SimpleSTLSTM(nn.Module):
    def __init__(self, n_stations, hidden_size, num_layers):
        super(SimpleSTLSTM, self).__init__()
        self.n_stations = n_stations
        self.hidden_size = hidden_size
        
        # Input size: 2 (In/Out) + 2 (Time) = 4
        # 这里简化处理，不加邻域卷积，直接对每个站点独立预测但共享参数，
        # 或者将所有站点作为特征输入 [S * 2] -> 太大
        # 考虑到站点间相关性，用简单的全连接层混合站点信息
        
        # 方案：输入 [Batch, Seq, Stations, 2+2] -> [Batch, Seq, Stations * 4]
        # LSTM 处理序列
        # 输出 [Batch, Seq, Hidden] -> [Batch, Stations * 2]
        
        self.input_size = n_stations * 2 + 2 # Flatten stations, plus global time
        
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, n_stations * 2)
        self.relu = nn.ReLU()
        
    def forward(self, x_flow, x_time):
        # x_flow: [B, L, S, 2]
        # x_time: [B, L, 2]
        B, L, S, _ = x_flow.size()
        
        # Flatten stations: [B, L, S*2]
        x_flow_flat = x_flow.view(B, L, -1)
        
        # Concat time: [B, L, S*2 + 2]
        x = torch.cat([x_flow_flat, x_time], dim=2)
        
        out, _ = self.lstm(x)
        # Take last step: [B, H]
        out = out[:, -1, :]
        
        pred = self.fc(out) # [B, S*2]
        pred = pred.view(B, 1, S, 2) # [B, 1, S, 2]
        
        # 保证非负 (虽然是对数空间预测，不需要非负，还原后exp即可，
        # 但如果直接预测数值需要ReLU。这里预测的是log值，可以为负)
        return pred

model = SimpleSTLSTM(n_stations, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 5. 训练
print("Starting training...")
best_val_loss = float('inf')
patience = 5
counter = 0

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for (x_flow, x_time), (y_flow, _) in train_loader:
        x_flow, x_time = x_flow.to(DEVICE), x_time.to(DEVICE)
        y_flow = y_flow.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(x_flow, x_time)
        loss = criterion(output, y_flow)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for (x_flow, x_time), (y_flow, _) in val_loader:
            x_flow, x_time = x_flow.to(DEVICE), x_time.to(DEVICE)
            y_flow = y_flow.to(DEVICE)
            output = model(x_flow, x_time)
            total_val_loss += criterion(output, y_flow).item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# 6. 预测 2016-09-01 07:00 - 09:00
print("Predicting for 2016-09-01...")
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
model.eval()

# 构造输入: 我们需要从已知数据的最后时刻开始滚动预测
# 数据集实际上只包含到 2016-08-31 23:58 的数据，因此最后完整的每小时数据是 2016-08-31 23:00
# 我们需要预测 09-01 00:00 到 09-01 08:00 (共9步)，取最后两步作为结果

last_data_time_str = "2016-08-31 23:00:00"
curr_idx = time_idx.get_loc(pd.Timestamp(last_data_time_str))
print(f"Starting rolling prediction from: {last_data_time_str}")

# 准备初始输入 [08-30 00:00 ... 08-31 23:00] (48 hours)
start_input_idx = curr_idx - SEQ_LEN + 1
input_seq_flow = flow_scaled[start_input_idx : curr_idx+1]
input_seq_time = time_features[start_input_idx : curr_idx+1]

current_flow = torch.FloatTensor(input_seq_flow).unsqueeze(0).to(DEVICE) # [1, 48, S, 2]
current_time = torch.FloatTensor(input_seq_time).unsqueeze(0).to(DEVICE) # [1, 48, 2]

predictions = []
# 预测 00:00 到 08:00 (共9步)
steps_to_predict = 9
start_pred_time = pd.Timestamp("2016-09-01 00:00:00")

for i in range(steps_to_predict):
    # Calculate target time for this step
    target_time = start_pred_time + pd.Timedelta(hours=i)
    target_hour = target_time.hour
    target_dow = target_time.dayofweek / 6.0
    
    with torch.no_grad():
        pred_next = model(current_flow, current_time) # [1, 1, S, 2]
    
    predictions.append(pred_next.cpu().numpy())
    
    # 滚动更新: 移除最早一个，加入预测值
    # 更新 flow
    current_flow = torch.cat([current_flow[:, 1:, :, :], pred_next], dim=1)
    
    # 更新 time (构造下一个时间步特征)
    next_time_feat = np.array([[np.sin(2 * np.pi * target_hour / 24.0), target_dow]])
    next_time_tensor = torch.FloatTensor(next_time_feat).unsqueeze(0).to(DEVICE) # [1, 1, 2]
    current_time = torch.cat([current_time[:, 1:, :], next_time_tensor], dim=1)

# 还原预测值
predictions = np.concatenate(predictions, axis=1) # [1, 9, S, 2]
# 我们只关心最后两个时间步 (07:00, 08:00) -> indices 7, 8
predictions = predictions[:, 7:9, :, :] # [1, 2, S, 2]

predictions = predictions.reshape(2 * n_stations, 2) # [2*S, 2]
predictions_inv = scaler.inverse_transform(predictions) # [2*S, 2]
predictions_inv = np.expm1(predictions_inv) # log1p inverse
predictions_inv = np.maximum(predictions_inv, 0) # ReLU
predictions_inv = predictions_inv.reshape(2, n_stations, 2) # [2, S, 2]

# 保存预测结果
# 格式: station_id, 07_in, 07_out, 08_in, 08_out
results = []
for i, sid in enumerate(top_50):
    row = {
        'station_id': sid,
        'pred_07_out': predictions_inv[0, i, 0],
        'pred_07_in': predictions_inv[0, i, 1],
        'pred_08_out': predictions_inv[1, i, 0],
        'pred_08_in': predictions_inv[1, i, 1]
    }
    results.append(row)

pred_df = pd.DataFrame(results)
pred_df.to_csv(os.path.join(OUTPUT_DIR, "q1_predictions.csv"), index=False)
print("Saved predictions to q1_predictions.csv")

# 训练曲线
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Training Loss')
plt.savefig(os.path.join(OUTPUT_DIR, "q1_training_loss.png"))
