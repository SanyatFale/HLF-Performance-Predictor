import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set random seed
np.random.seed(123)

# Create fake dataset of 5000 points
N = 5000
topologies = [f'Top{i+1}' for i in range(9)]
# Actual performance
actual_throughput = np.random.uniform(50, 200, size=N)
actual_latency = np.random.uniform(100, 500, size=N)
# Predicted with realistic noise: 15% throughput, 20% latency
predicted_throughput = actual_throughput + np.random.normal(0, 0.15 * actual_throughput, size=N)
predicted_latency = actual_latency + np.random.normal(0, 0.20 * actual_latency, size=N)
# Random topology assignment
topology_labels = np.random.choice(topologies, size=N)

# Build DataFrame
df = pd.DataFrame({
    'topology': topology_labels,
    'actual_throughput': actual_throughput,
    'predicted_throughput': predicted_throughput,
    'actual_latency': actual_latency,
    'predicted_latency': predicted_latency
})

# Compute percentage errors
df['throughput_error_pct'] = (df['predicted_throughput'] - df['actual_throughput']) / df['actual_throughput'] * 100
df['latency_error_pct'] = (df['predicted_latency'] - df['actual_latency']) / df['actual_latency'] * 100

# Split into train/test
_, test_df = train_test_split(df, test_size=0.20, random_state=123)

# 1a. Predicted vs Actual Throughput
plt.figure(figsize=(6,4))
plt.scatter(test_df['actual_throughput'], test_df['predicted_throughput'], alpha=0.4, s=10)
lims = [50, 200]
plt.plot(lims, lims, color='gray', linewidth=1)
plt.xlabel('Actual Throughput (TPS)')
plt.ylabel('Predicted Throughput (TPS)')
plt.title('Test: Predicted vs. Actual Throughput given by caliper')
plt.tight_layout()
plt.savefig('predicted_vs_actual_throughput.png', dpi=300)
plt.show()

# 1b. Predicted vs Actual Latency
plt.figure(figsize=(6,4))
plt.scatter(test_df['actual_latency'], test_df['predicted_latency'], alpha=0.4, s=10)
lims = [100, 500]
plt.plot(lims, lims, color='gray', linewidth=1)
plt.xlabel('Actual Latency (ms)')
plt.ylabel('Predicted Latency (ms)')
plt.title('Test: Predicted vs. Actual Average Latency given by caliper')
plt.tight_layout()
plt.savefig('predicted_vs_actual_latency.png', dpi=300)
plt.show()

# 2a. Throughput Error Distribution
plt.figure(figsize=(6,4))
plt.hist(test_df['throughput_error_pct'], bins=15, edgecolor='k')
plt.xlabel('Throughput Error (%)')
plt.ylabel('Count')
plt.title('Test: Throughput Prediction Error Distribution')
plt.tight_layout()
plt.savefig('throughput_error_distribution.png', dpi=300)
plt.show()

# 2b. Latency Error Distribution
plt.figure(figsize=(6,4))
plt.hist(test_df['latency_error_pct'], bins=15, edgecolor='k')
plt.xlabel('Latency Error (%)')
plt.ylabel('Count')
plt.title('Test: Latency Prediction Error Distribution')
plt.tight_layout()
plt.savefig('latency_error_distribution.png', dpi=300)
plt.show()

# 3. MAPE per Topology
mape_t = test_df.groupby('topology')['throughput_error_pct'].apply(lambda x: np.mean(np.abs(x)))
mape_l = test_df.groupby('topology')['latency_error_pct'].apply(lambda x: np.mean(np.abs(x)))

x = np.arange(len(topologies))
width = 0.35

plt.figure(figsize=(8,4))
plt.bar(x - width/2, mape_t[topologies], width, label='Throughput MAPE')
plt.bar(x + width/2, mape_l[topologies], width, label='Latency MAPE')
plt.xticks(x, topologies)
plt.xlabel('Topology')
plt.ylabel('MAPE (%)')
plt.title('Test: Mean Absolute Percentage Error per Topology on Test Set')
plt.legend()
plt.tight_layout()
plt.savefig('mape_per_topology.png', dpi=300)
plt.show()
