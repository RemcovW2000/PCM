# --------------------
# Kozeny-Carman permeability calculation
# --------------------

# Given parameters
R = 7e-6       # filament radius in meters
k = 5          # Kozeny constant
Vf = 0.4       # fiber volume fraction

# Kozeny-Carman equation
K = (R**2 / (4 * k)) * ((1 - Vf)**3 / Vf**2)

print(f"Permeability K = {K:.4e} m^2")
