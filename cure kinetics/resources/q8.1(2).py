import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ----------------------------
# Given parameters
# ----------------------------
R = 7e-6           # meters, fibre radius
k = 5              # Kozeny constant

# Laminate parameters
AW = 0.4           # kg/m^2, areal weight
n_layers = 5       # number of layers
rho_f = 1800       # kg/m^3, fibre density
thickness_mm = np.array([1.76, 2.02, 2.30])  # mm
t_values = thickness_mm * 1e-3  # convert to meters

# Compute fibre volume fractions Vf = AW * n / (rho_f * t)
Vf_regions = AW * n_layers / (rho_f * t_values)

# Fibre volume fractions for nominal curve
Vf = np.arange(0.35, 0.66, 0.05)

# ----------------------------
# Permeability calculation
# ----------------------------
def kozeny_carman(Vf, R=R, k=k):
    return (R**2 / (4*k)) * ((1 - Vf)**3 / Vf**2)

K_nom = kozeny_carman(Vf)

# ----------------------------
# Plotting
# ----------------------------
fig, ax = plt.subplots(figsize=(8,5))

# Nominal K curve
ax.plot(Vf, K_nom, marker='o', linestyle='-', color='blue', label='Nominal K (Vf 0.35-0.65)')

# Indicate the three Vf points for the mould cavity
for Vf_r, t_r, color in zip(Vf_regions, thickness_mm, ['red', 'green', 'orange']):
    ax.axvline(Vf_r, color=color, linestyle='--', linewidth=1)
    ax.text(Vf_r+0.002, 1.2e-12, f'{t_r} mm', rotation=90, color=color, fontsize=10, verticalalignment='bottom')

# Logarithmic scale
ax.set_yscale('log')
ax.set_xlabel('Fiber volume fraction Vf', fontsize=12)
ax.set_ylabel('Permeability K (m²)', fontsize=12)
ax.set_title('Permeability vs Fiber Volume Fraction\nwith Mould Cavity Indications', fontsize=14)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2e}'))

# Legend
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
