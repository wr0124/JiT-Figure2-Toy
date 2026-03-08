import numpy as np
import matplotlib.pyplot as plt

# File path
path = "/data1/juliew/video_ref/JIT/toy_experiment/toy_highdim/outputs/all_signal_stats.txt"

# Output directory
outdir = "/data1/juliew/video_ref/JIT/toy_experiment/toy_highdim/"

# Load data (skip header)
data = np.loadtxt(path, skiprows=1)

rho = data[:,0]
D = data[:,1]
gamma = data[:,2]

# Remove D=2
mask = D != 2
rho = rho[mask]
D = D[mask]
gamma = gamma[mask]

unique_D = sorted(set(D))

# -------------------------
# Plot 1: gamma vs rho
# -------------------------
plt.figure(figsize=(6,4))

for d in unique_D:
    idx = D == d
    plt.plot(rho[idx], gamma[idx], marker="o", label=f"D={int(d)}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("rho")
plt.ylabel("gamma_eff")
plt.title("gamma_eff vs rho")
plt.legend()
plt.tight_layout()

plt.savefig(outdir + "gamma_vs_rho.png", dpi=300)
plt.close()


# -------------------------
# Plot 2: scaling collapse
# -------------------------
plt.figure(figsize=(6,4))

gamma_scaled = gamma * np.sqrt(D)

for d in unique_D:
    idx = D == d
    plt.plot(rho[idx], gamma_scaled[idx], marker="o", label=f"D={int(d)}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("rho")
plt.ylabel("gamma_eff * sqrt(D)")
plt.title("Scaling collapse")
plt.legend()
plt.tight_layout()

plt.savefig(outdir + "gamma_scaling_collapse.png", dpi=300)
plt.close()
