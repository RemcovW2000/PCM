import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Given parameters
# ----------------------------
L = 1.0  # m, total flow length
A = 0.25  # m^2, cross-sectional area
K = 3.31e-12  # m^2, permeability
t_total = 90  # min, gelation time
eta0 = 0.03  # Pa.s, initial viscosity
eta_final = 0.045  # Pa.s, final viscosity

# Mold volume and total flow rate
V_mold = A * L  # m^3
t_sec = t_total * 60  # s
Q = V_mold / t_sec  # m^3/s

# ----------------------------
# Time array
# ----------------------------
t = np.linspace(0, t_total, 100)

# Viscosity over time
eta = eta0 + (eta_final - eta0) * (t / t_total)

# ----------------------------
# Number of inlets to compare
# ----------------------------
inlets = [1, 2, 4]  # Example: 1, 2, and 4 inlets
colors = ['blue', 'green', 'red']

plt.figure(figsize=(8, 5))

for i, n in enumerate(inlets):
    L_eff = L / n  # effective flow length for each inlet
    dP_dL = eta * Q / (K * A)  # pressure gradient [Pa/m]
    dP = dP_dL * L_eff  # total pressure per inlet [Pa]
    dP_bar = dP / 1e5  # convert to Bar

    plt.plot(t, dP_bar, color=colors[i], linewidth=2, label=f'{n} inlet(s)')

# ----------------------------
# Plot formatting
# ----------------------------
plt.xlabel("Time [min]")
plt.ylabel("Injection Pressure [Bar]")
plt.title("Effect of Number of Inlets on Injection Pressure")
plt.grid(True)
plt.legend()
plt.show()
