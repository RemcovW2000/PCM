import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Given parameters
# ----------------------------
R = 7e-6           # meters, fibre radius
k = 5              # Kozeny constant
eta = 0.03         # Pa.s, resin viscosity
L = 1.0            # m, flow distance
t_infusion = 40*60 # seconds

# Laminate parameters for Vf calculation
AW = 0.4           # kg/m^2
n_layers = 5
rho_f = 1800       # kg/m^3
thickness_mm = np.array([1.76, 2.02, 2.30])  # mm
t_values = thickness_mm * 1e-3

# Compute Vf for the three regions
Vf_regions = AW * n_layers / (rho_f * t_values)

# Fibre volume fractions for nominal curve
Vf = np.arange(0.35, 0.66, 0.05)

# ----------------------------
# Permeability function
# ----------------------------
def kozeny_carman(Vf, R=R, k=k):
    return (R** 2 / (4*k)) * ((1 - Vf)** 3 / Vf **2)

K_nom = kozeny_carman(Vf)

# Required pressure: ΔP = (L^2 * η) / (K * t)
deltaP_nom = (L**2 * eta) / (K_nom * t_infusion)  # in Pa

# ----------------------------
# Plotting ΔP vs Vf
# ----------------------------
fig, ax = plt.subplots(figsize=(8,5))

# Nominal ΔP curve
ax.plot(Vf, deltaP_nom/1e5, marker='o', linestyle='-', color='blue', label='ΔP vs Vf (Nominal)')

# Indicate the three Vf points for the mould cavity
colors = ['red', 'green', 'orange']
for Vf_r, t_r, color in zip(Vf_regions, thickness_mm, colors):
    K_r = kozeny_carman(Vf_r)
    deltaP_r = (L**2 * eta) / (K_r * t_infusion) / 1e5  # convert to bar
    ax.plot(Vf_r, deltaP_r, marker='x', markersize=10, color=color)
    ax.text(Vf_r + 0.002, deltaP_r*1.05, f'{t_r} mm', color=color, fontsize=10)

# Labels, title, grid, legend
ax.set_xlabel('Fiber volume fraction Vf', fontsize=12)
ax.set_ylabel('Required Injection Pressure ΔP (bar)', fontsize=12)
ax.set_title('Required Pressure vs Fiber Volume Fraction\nwith Mould Cavity Indications', fontsize=14)
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
