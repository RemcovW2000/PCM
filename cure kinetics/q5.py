# find d halpha/ dt ad alpha = 0 for all three datasets
import numpy as np
from matplotlib import pyplot as plt

from q1 import final_data_120, final_data_150, final_data_180

from q3 import fraction_cured_120, fraction_cured_150, fraction_cured_180

dalpha_dt_120 = np.gradient(fraction_cured_120, final_data_120['Time'])
dalpha_dt_150 = np.gradient(fraction_cured_150, final_data_150['Time'])
dalpha_dt_180 = np.gradient(fraction_cured_180, final_data_180['Time'])

R = 8.314  # J/(mol·K)

d_vals = [dalpha_dt_120[0], dalpha_dt_150[0], dalpha_dt_180[0]]
T_vals = np.array([120, 150, 180]) + 273.15
# linearize and fit: ln(d) = -E1 * (1/(R*T)) + ln(A1)
x = 1.0 / (R * T_vals)
y = np.log(d_vals)
m, b = np.polyfit(x, y, 1)

E1 = -m
A1 = np.exp(b)

def k1(A1, E1, T):
    return A1 * np.exp(-E1 / (R * T))

if __name__ == "__main__":
    print((k1(A1, E1, T_vals[0]) - dalpha_dt_120[0]) / dalpha_dt_120[0])
    print((k1(A1, E1, T_vals[1]) - dalpha_dt_150[0]) / dalpha_dt_150[0])
    print((k1(A1, E1, T_vals[2]) - dalpha_dt_180[0]) / dalpha_dt_180[0])
