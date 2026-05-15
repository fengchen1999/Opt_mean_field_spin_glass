import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 24,
    "font.family": "Times New Roman"
})

# Problem dimensions
dims = np.array([5, 100, 200, 500, 1000])

# Mean and standard deviation from your results
ising_mean = np.array([1.2058, 1.4394, 1.4910, 1.5027, 1.5084])
ising_std  = np.array([0.5583, 0.0337, 0.0195, 0.0100, 0.0050])

spectral_mean = np.array([1.5734, 1.9126, 1.9744, 1.9837, 1.9848])
spectral_std  = np.array([0.5656, 0.0487, 0.0329, 0.0204, 0.0152])

sdp_mean = np.array([1.2188, 1.6987, 1.7847, 1.8417, 1.8815])
sdp_std  = np.array([0.5479, 0.0307, 0.0200, 0.0089, 0.0045])

plt.figure(figsize=(9, 6))

plt.errorbar(
    dims, ising_mean, yerr=ising_std,
    marker="o", capsize=4, linewidth=2,
    label="Ising"
)

plt.errorbar(
    dims, spectral_mean, yerr=spectral_std,
    marker="s", capsize=4, linewidth=2,
    label="Spectral"
)

plt.errorbar(
    dims, sdp_mean, yerr=sdp_std,
    marker="^", capsize=4, linewidth=2,
    label="SDP"
)

# Theoretical optimal value for Ising, normalized by N
plt.axhline(
    y=1.526,
    linestyle="--",
    linewidth=2.5,
    color="black",
    label="Theoretical optimum"
)

plt.xlabel("Problem dimension $N$")
plt.ylabel("Mean objective value")

plt.xticks(dims, dims)
plt.grid(True, alpha=0.4)
plt.legend(fontsize=18)

plt.tight_layout()
plt.savefig(
    "./Results/relax_comp.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()