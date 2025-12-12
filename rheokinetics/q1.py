import numpy as np
import matplotlib.pyplot as plt

class CastroMacosko:
    def __init__(self, alpha_g: float, c1: float, c2: float, T_b: float, A: float):
        self.alpha_g = alpha_g
        self.c1 = c1
        self.c2 = c2
        self.T_b = T_b
        self.A = A

    def visc(self, alpha, temperature) -> float:
        term_1 = self.A * np.exp(self.T_b/temperature)
        term_2 = (self.alpha_g / (self.alpha_g - alpha))**(self.c1 + self.c2 * alpha)
        return term_1 * term_2

arocy_l_10 = CastroMacosko(
    alpha_g=0.64,
    c1=2.32,
    c2=1.4,
    T_b=5160.39,
    A=3.32e-8
)

alpha_list = np.linspace(0.00, 1.0, 1000)
temperature_list = [t + 273.15 for t in [120, 150, 180]]

viscs_data = {}
for temperature in temperature_list:
    viscosity_list = [arocy_l_10.visc(alpha, temperature) for alpha in alpha_list]
    viscs_data[temperature] = viscosity_list

plt.plot(alpha_list, viscs_data[temperature_list[0]], label=f'T={temperature_list[0]-273.15}°C')
plt.plot(alpha_list, viscs_data[temperature_list[1]], label=f'T={temperature_list[1]-273.15}°C')
plt.plot(alpha_list, viscs_data[temperature_list[2]], label=f'T={temperature_list[2]-273.15}°C')
plt.ylim(1e-3, 1e3)
plt.yscale('log')
plt.xlabel('Degree of Cure (α)')
plt.ylabel('Viscosity (Pa.s)')
plt.title('Viscosity vs Degree of Cure at Different Temperatures')
plt.legend()
plt.show()