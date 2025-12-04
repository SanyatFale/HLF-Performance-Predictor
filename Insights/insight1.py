import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Plot 1: Throughput vs. Number of Peers (with overhead effects)
peers = np.array([2, 4, 6, 8, 10, 12])
throughput = 150 - 2 * peers + np.random.normal(0, 5, size=peers.shape)  # slight decrease due to comm. overhead
throughput[throughput < 80] = 80 + np.random.normal(0, 3)  # floor effect

plt.figure(figsize=(6,4))
plt.plot(peers, throughput, marker='o')
plt.xlabel("Number of Peers")
plt.ylabel("Throughput (TPS)")
plt.title("Throughput vs. Number of Peers")
plt.grid(True)
plt.tight_layout()
plt.savefig("throughput_vs_peers.png")
plt.show()

# Plot 3: Throughput vs. Block Size (with saturation effects)
block_sizes = np.array([50, 100, 200, 400, 800, 1200])
throughput_bs = 50 + 0.2 * block_sizes - 0.00005 * block_sizes**2 + np.random.normal(0, 5, size=block_sizes.shape)

plt.figure(figsize=(6,4))
plt.plot(block_sizes, throughput_bs, marker='o', color='green')
plt.xlabel("Block Size (txs per block)")
plt.ylabel("Throughput (TPS)")
plt.title("Throughput vs. Block Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("throughput_vs_block_size.png")
plt.show()

# Plot 4: Latency vs. Number of Orderers (with consensus delay effect)
orderers = np.array([1, 2, 3, 4, 5])
latency = 200 - 10 * orderers + 3 * orderers**2 + np.random.normal(0, 15, size=orderers.shape)

plt.figure(figsize=(6,4))
plt.plot(orderers, latency, marker='o', color='red')
plt.xlabel("Number of Orderers")
plt.ylabel("Latency (ms)")
plt.title("Latency vs. Number of Orderers")
plt.grid(True)
plt.tight_layout()
plt.savefig("latency_vs_orderers.png")
plt.show()
