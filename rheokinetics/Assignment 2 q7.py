import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1             # m
K = 3.3075e-12      # m^2
eta0 = 0.03         # Pa.s
eta_end = 0.045     # Pa.s
t_total = 90        # min

t = np.linspace(1, t_total, 100)  # start from 1 min to avoid division by zero
eta_t = eta0 + (eta_end - eta0)/t_total * t

dP = (L**2 * eta_t) / (K * t * 60)  # Pa
dP_bar = dP / 1e5

plt.plot(t, dP_bar)
plt.xlabel('Time [min]')
plt.ylabel('Pressure difference [Bar]')
plt.title('Required pressure difference vs time for constant flow')
plt.grid(True)
plt.show()
