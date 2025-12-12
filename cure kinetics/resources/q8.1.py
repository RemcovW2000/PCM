import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

# Given parameters
R = 7e-6  # meters
k = 5

# Volume fractions
Vf = np.arange(0.35, 0.66, 0.05)

# Permeability calculation
K = (R**2 / (4*k)) * ((1 - Vf)**3 / Vf**2)

# Plot
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(Vf, K, marker='o', linestyle='-')

# Logarithmic scale for spacing
ax.set_yscale('log')

# Axis labels and title
ax.set_xlabel('Fiber volume fraction Vf', fontsize=12)
ax.set_ylabel('Permeability K (m²)', fontsize=12)
ax.set_title('Permeability vs Fiber Volume Fraction', fontsize=14)

# Gridlines for both major and minor ticks
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Set major ticks automatically and format in scientific notation
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=np.arange(1, 10)*0.1, numticks=10))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2e}'))

plt.tight_layout()
plt.show()
