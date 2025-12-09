import numpy as np
from scipy.optimize import least_squares
from q5 import (
    A1_solution, E1_solution,
    fraction_cured_120, dalpha_dt_120,
    fraction_cured_150, dalpha_dt_150,
    fraction_cured_180, dalpha_dt_180
)

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
    150 + 273.15: interp_rate(fraction_cured_150, dalpha_dt_150, alpha_vals),
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
