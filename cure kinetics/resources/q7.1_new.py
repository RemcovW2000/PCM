import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Given parameters
# ----------------------------
L = 1.0            # m, flow length
A = 0.25           # m^2, cross-sectional area
K = 3.31e-12       # m^2, permeability
t_total = 90       # min, gelation time
eta0 = 0.03        # Pa.s, initial viscosity
eta_final = 0.045  # Pa.s, final viscosity

# Calculate the flow rate Q from mold volume / gelation time
V_mold = A * L              # m^3, mold volume
t_sec = t_total * 60         # s, gelation time in seconds
Q = V_mold / t_sec           # m^3/s, constant flow rate

# ----------------------------
# Time array
# ----------------------------
t = np.linspace(0, t_total, 100)  # minutes

# ----------------------------
# Viscosity as a function of time
# ----------------------------
eta = eta0 + (eta_final - eta0) * (t / t_total)

# ----------------------------
# Pressure gradient over time
# ----------------------------
dP_dL = eta * Q / (K * A)      # Pa/m
dP_dL_bar = dP_dL / 1e5        # convert Pa to Bar/m

# ----------------------------
# Plotting
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(t, dP_dL_bar, color='blue', linewidth=2)
plt.xlabel("Time [min]")
plt.ylabel("Pressure gradient [Bar/m]")
plt.title("Required Pressure Gradient over Time")
plt.grid(True)
plt.show()
