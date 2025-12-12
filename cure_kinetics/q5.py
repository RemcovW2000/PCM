# find d halpha/ dt ad alpha = 0 for all three datasets
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from q1 import final_data_120, final_data_150, final_data_180

from q3 import fraction_cured_120, fraction_cured_150, fraction_cured_180

dalpha_dt_120 = np.gradient(fraction_cured_120, final_data_120['Time Seconds'])
dalpha_dt_150 = np.gradient(fraction_cured_150, final_data_150['Time Seconds'])
dalpha_dt_180 = np.gradient(fraction_cured_180, final_data_180['Time Seconds'])

R = 8.314  # J/(mol·K)

d_vals = [dalpha_dt_120[0], dalpha_dt_150[0]]
T_vals = np.array([120, 150]) + 273.15

# linearize and and use least squares to fit: ln(d) = -E1 * (1/(R*T)) + ln(A1_solution)
x = 1.0 / (R * T_vals)
y = np.log(d_vals)
m, b = np.polyfit(x, y, 1)

E1_solution = -m
A1_solution = np.exp(b)

def k1(A1, E1, T):
    return A1 * np.exp(-E1 / (R * T))

if __name__ == "__main__":
    print("Error at 120 deg: ", (k1(A1_solution, E1_solution, T_vals[0]) - dalpha_dt_120[0]) / dalpha_dt_120[0])
    print("Error at 150 deg: ", (k1(A1_solution, E1_solution, T_vals[1]) - dalpha_dt_150[0]) / dalpha_dt_150[0])
    print("Error at 180 deg: ", (k1(A1_solution, E1_solution, 180 + 273.15) - dalpha_dt_180[0]) / dalpha_dt_180[0])
    print("E1: ", E1_solution)
    print("A1_solution: ", A1_solution)


    index_at_low_alpha_120 = next(
            i for i, v in enumerate(fraction_cured_120) if v >= 0.01)
    index_at_low_alpha_150 = next(
        i for i, v in enumerate(fraction_cured_150) if v >= 0.01)
    index_at_low_alpha_180 = next(
        i for i, v in enumerate(fraction_cured_180) if v >= 0.01)

    ln_dalpha_dt = [
        np.log(dalpha_dt_120[index_at_low_alpha_120]),
        np.log(dalpha_dt_150[index_at_low_alpha_150]),
    ]
    fig, ax = plt.subplots()

    ax.plot(1/np.asarray(T_vals), ln_dalpha_dt, 'o-')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    plt.xlabel('1 / T (1/K)')
    plt.ylabel('ln(dα/dt) at α=0.01')
    plt.title('Curing Rate at Low Degree of Cure, at 120, 150 °C')
    plt.show()