import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# === Load model metadata ===
metadata = torch.load("/Users/adityamanjunatha/Desktop/SSDS_Repo/model_metadata.pth")
feature_cols = metadata["feature_cols"]
workload_map = metadata["workload_map"]
input_dim = len(feature_cols)

# === Define the model architecture ===
class SimpleRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# === Load trained models ===
model_th = SimpleRegressor(input_dim)
model_th.load_state_dict(torch.load("/Users/adityamanjunatha/Desktop/SSDS_Repo/best_throughput_model.pth"))
model_th.eval()

model_lt = SimpleRegressor(input_dim)
model_lt.load_state_dict(torch.load("/Users/adityamanjunatha/Desktop/SSDS_Repo/best_latency_model.pth"))
model_lt.eval()

# === Recompute StandardScaler from training data ===
df = pd.read_csv("fabric__dataset.csv")  # use your actual training CSV
df["workload_enc"] = df["workload"].map(workload_map)
X = df[feature_cols].values.astype(float)
scaler = StandardScaler().fit(X)

# === Sample input (change this as needed) ===
sample = {
    "workload": "mixed",
    "workers": 2,
    "tps": 50,
    "txs": 1000
}

# === Prepare input vector ===
x_raw = [
    workload_map[sample["workload"]],
    sample["workers"],
    sample["tps"],
    sample["txs"]
]
x_scaled = scaler.transform([x_raw])
x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

# === Inference ===
with torch.no_grad():
    pred_throughput = model_th(x_tensor).item()
    pred_latency = model_lt(x_tensor).item()

# === Output ===
print("\n📊 Inference Results")
print(f"Input: {sample}")
print(f"Predicted Throughput (TPS): {pred_throughput:.2f}")
print(f"Predicted Latency (ms):     {pred_latency:.2f}")
