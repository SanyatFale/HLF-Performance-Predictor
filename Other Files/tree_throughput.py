import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load data
csv_path = "/Users/adityamanjunatha/Desktop/SSDS_Repo/fabric__dataset.csv"  # adjust if needed
df = pd.read_csv(csv_path)

# 2. Encode the workload feature (if not already encoded)
df["workload_enc"] = df["workload"].map({
    "createCar": 0,
    "queryCar" : 1,
    "mixed"    : 2
})

# 3. Select features and target
feature_cols = ["workload_enc", "workers", "tps", "txs"]
X = df[feature_cols].values
y = df["throughput"].values

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create DMatrix objects (optional)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_cols)

# 6. Parameter setup
params = {
    "objective":      "reg:squarederror",
    "eval_metric":    "rmse",
    "max_depth":      4,
    "eta" :           0.1,
    "seed":           42,
}
num_round = 100

# 7. Train with evaluation on both train and test
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_round,
    evals=[(dtrain, "train"), (dtest, "eval")],
    verbose_eval=False
)

# 8. Predict and evaluate
y_pred = bst.predict(dtest)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("XGBoost Throughput Regression Results:")
print(f"  MSE: {mse:.2f}")
print(f"  MAE: {mae:.2f}")
print(f"  R² : {r2:.3f}")

# 9. Save the model
bst.save_model("xgb_throughput_model.json")
print("Model saved to xgb_throughput_model.json")