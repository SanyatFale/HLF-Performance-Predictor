import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
csv_path = "/Users/adityamanjunatha/Desktop/SSDS_Repo/fabric__dataset.csv"
df = pd.read_csv(csv_path)
df["workload_enc"] = df["workload"].map({"createCar": 0, "queryCar": 1, "mixed": 2})

X = df[["workload_enc", "workers", "tps", "txs"]].values.astype(float)
y_th = df["throughput"].values.astype(float).reshape(-1,1)
y_lt = df["avg_latency"].values.astype(float).reshape(-1,1)

# Train/Validation split
X_train, X_val, y_th_train, y_th_val, y_lt_train, y_lt_val = train_test_split(
    X, y_th, y_lt, test_size=0.2, random_state=42
)

# Normalize features
scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
X_val   = scaler_X.transform(X_val)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_th_train_t = torch.tensor(y_th_train, dtype=torch.float32)
y_th_val_t   = torch.tensor(y_th_val,   dtype=torch.float32)
y_lt_train_t = torch.tensor(y_lt_train, dtype=torch.float32)
y_lt_val_t   = torch.tensor(y_lt_val,   dtype=torch.float32)

# DataLoaders
batch_size = 16
train_loader_th = DataLoader(TensorDataset(X_train_t, y_th_train_t),
                             batch_size=batch_size, shuffle=True)
train_loader_lt = DataLoader(TensorDataset(X_train_t, y_lt_train_t),
                             batch_size=batch_size, shuffle=True)

# Model definition
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

# Instantiate models
input_dim = X_train.shape[1]
model_th = SimpleRegressor(input_dim)
model_lt = SimpleRegressor(input_dim)

# Loss functions and optimizers
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
opt_th = torch.optim.Adam(model_th.parameters(), lr=1e-3)
opt_lt = torch.optim.Adam(model_lt.parameters(), lr=1e-3)

# Training with checkpointing
best_th_loss = float('inf')
best_lt_loss = float('inf')
epochs = 1000

for epoch in range(1, epochs+1):
    model_th.train(); model_lt.train()
    for (x_batch, y_th_batch), (_, y_lt_batch) in zip(train_loader_th, train_loader_lt):
        # Throughput model update
        opt_th.zero_grad()
        loss_th = criterion_mse(model_th(x_batch), y_th_batch)
        loss_th.backward(); opt_th.step()
        # Latency model update (first 80 epochs)
        if epoch <= 80:
            opt_lt.zero_grad()
            loss_lt = criterion_mse(model_lt(x_batch), y_lt_batch)
            loss_lt.backward(); opt_lt.step()
    
    # Validation phase
    model_th.eval(); model_lt.eval()
    with torch.no_grad():
        th_preds = model_th(X_val_t)
        lt_preds = model_lt(X_val_t)
        val_mse_th = criterion_mse(th_preds, y_th_val_t).item()
        val_mae_th = criterion_mae(th_preds, y_th_val_t).item()
        val_mse_lt = criterion_mse(lt_preds, y_lt_val_t).item()
        val_mae_lt = criterion_mae(lt_preds, y_lt_val_t).item()
    
    # Checkpoint best models
    if val_mse_th < best_th_loss:
        best_th_loss = val_mse_th
        torch.save(model_th.state_dict(), "best_throughput_model.pth")
    if val_mse_lt < best_lt_loss:
        best_lt_loss = val_mse_lt
        torch.save(model_lt.state_dict(), "best_latency_model.pth")
    
    # Logging
    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{epochs} | "
              f"Val Th MSE: {val_mse_th:.4f}, Th MAE: {val_mae_th:.4f} | "
              f"Val Lt MSE: {val_mse_lt:.4f}, Lt MAE: {val_mae_lt:.4f}")

print("Training complete.")
print(f"Best validation Throughput MSE: {best_th_loss:.4f}")
print(f"Best validation Latency MSE: {best_lt_loss:.4f}")

# Save final configuration metadata
torch.save({
    'workload_map': {"createCar": 0, "queryCar": 1, "mixed": 2},
    'feature_cols': ["workload_enc", "workers", "tps", "txs"]
}, "model_metadata.pth")
