from scipy.optimize import least_squares

from cure_kinetics.q3 import cure_rate_120, cure_rate_180
from q5 import (
    A1_solution, E1_solution,
    fraction_cured_120, dalpha_dt_120,
    fraction_cured_150, dalpha_dt_150,
    fraction_cured_180, dalpha_dt_180
)

import numpy as np
from resources.constants import START_TIME_150, START_TIME_120
from q1 import final_data_150, final_data_120, final_data_180
from q3 import fraction_cured_150, fraction_cured_120, fraction_cured_180
from q5 import E1_solution, A1_solution

R = 8.314

def k1(T):
    return A1_solution * np.exp(-E1_solution / (R * T))

def da_dt(A2, E2, m, n, alpha, T):
    term1 = k1(T)
    term2 = A2 * np.exp(-E2 / (R * T)) * (alpha ** m)
    return (term1 + term2) * ((1 - alpha) ** n)

def interp_rate(frac, rate, alpha_vals):
    """Interpolation helper: get rate(alpha)."""
    return np.interp(alpha_vals, frac, rate)


# --- Construct dataset at common alpha values ---
alpha_vals = np.linspace(0.05, 0.95, 400)

data = {
    120 + 273.15: interp_rate(fraction_cured_120, dalpha_dt_120, alpha_vals),
    180 + 273.15: interp_rate(fraction_cured_180, dalpha_dt_180, alpha_vals),
}


# --- Residual function for least_squares ---
def residuals(params):
    log_A2, E2, m, n = params
    A2 = 10 ** log_A2

    res = []

    for T, measured_rates in data.items():
        predicted = da_dt(A2, E2, m, n, alpha_vals, T)
        res.extend(predicted - measured_rates)

    return np.array(res)


# --- Initial guess ---
x0 = np.array([6.0, 6e4, 1.2, 1.25])  # log_A2, E2, m, n

# --- Bounds in *physical space* ---
lower_bounds = [3.0, 3e4, 0.1, 0.1]   # log10(A2), E2, m, n
upper_bounds = [10.0, 2e5, 3.0, 3.0]


# --- Run optimizer ---
result = least_squares(
    residuals,
    x0,
    bounds=(lower_bounds, upper_bounds),
    method="trf",
    ftol=1e-12,
    xtol=1e-12,
    gtol=1e-12,
    max_nfev=20000
)

log_A2_solution, E2_solution, m_solution, n_solution = result.x
A2_solution = 10 ** log_A2_solution

print("A2 =", A2_solution)
print("E2 =", E2_solution)
print("m  =", m_solution)
print("n  =", n_solution)

def da_dt(A1, E1, A2, E2, m, n, alpha, T):
    R = 8.314  # J/(mol·K)
    term_1 = A1 * np.exp(-E1 / (R * T))
    term_2 = A2 * np.exp(-E2 / (R * T)) * (alpha ** m)
    return (term_1 + term_2) * ((1 - alpha) ** n)

def simulate_cure(A1, E1, A2, E2, m, n, T_C, t_lst):
    T = T_C + 273.15
    alpha = 0.0
    alpha_vals = [0.0]
    da_dt_vals = []
    for i in range(1, len(t_lst)):
        dt = t_lst[i] - t_lst[i-1]
        dadt = da_dt(A1, E1, A2, E2, m, n, alpha, T)
        alpha += dadt * dt

        da_dt_vals.append(dadt)
        alpha_vals.append(alpha)
    return np.array(da_dt_vals)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # Example parameters
    t_lst = np.linspace(0, 30000, 1000000)

    for cure_temp in [120, 180]:
        sim_results = simulate_cure(A1_solution, E1_solution, A2_solution, E2_solution, m_solution, n_solution, cure_temp, t_lst)
        plt.plot(t_lst[:-1], sim_results, label=f'Simulated {cure_temp}°C')

    plt.plot([t - START_TIME_120*60 for t in final_data_120["Time Seconds"]], cure_rate_120, '--', label='Experimental 120°C')
    plt.plot([t -2.5*60 for t in final_data_180["Time Seconds"]], cure_rate_180, '--', label='Experimental 180°C')
    plt.xlabel('Time (s)')
    plt.ylabel('Degree of Cure (α)')
    plt.title('Cure rate vs. Time, simulated vs. experimental')
    plt.xlim(0, 5000)
    plt.axhline(0, color='k', linestyle='--')
    plt.legend()
    plt.show()