import numpy as np
from scipy.optimize import minimize
from q5 import A1_solution, fraction_cured_120, fraction_cured_150, fraction_cured_180, \
    dalpha_dt_120, dalpha_dt_150, dalpha_dt_180, E1_solution


# Parameter bounds (physical)
BOUNDS = {
    'log_A2': (1, 10),      # log10(A2): 1e3 to 1e10
    'E2': (1e4, 2e10),       # J/mol
    'm': (0.1, 6),
    'n': (0.1, 2.5)
}


def normalize(value, bounds):
    """Map value from [low, high] to [0, 1]."""
    low, high = bounds
    return (value - low) / (high - low)


def denormalize(norm_value, bounds):
    """Map value from [0, 1] to [low, high]."""
    low, high = bounds
    return norm_value * (high - low) + low


def params_to_normalized(log_A2, E2, m, n):
    """Convert physical parameters to normalized [0,1] space."""
    return [
        normalize(log_A2, BOUNDS['log_A2']),
        normalize(E2, BOUNDS['E2']),
        normalize(m, BOUNDS['m']),
        normalize(n, BOUNDS['n'])
    ]


def normalized_to_params(norm_params):
    """Convert normalized [0,1] parameters to physical values."""
    log_A2 = denormalize(norm_params[0], BOUNDS['log_A2'])
    E2 = denormalize(norm_params[1], BOUNDS['E2'])
    m = denormalize(norm_params[2], BOUNDS['m'])
    n = denormalize(norm_params[3], BOUNDS['n'])
    A2 = 10 ** log_A2
    return A2, E2, m, n


def da_dt(A2, E2, m, n, alpha, T):
    R = 8.314  # J/(mol·K)
    term_1 = A1_solution * np.exp(-E1_solution / (R * T))
    term_2 = A2 * np.exp(-E2 / (R * T)) * (alpha ** m)
    return (term_1 + term_2) * ((1 - alpha) ** n)


def calculate_error_normalized(norm_params, dalpha_dt_vals, alpha_vals):
    """Calculate SSE using normalized parameters."""
    A2, E2, m, n = normalized_to_params(norm_params)
    error = 0.0
    for temp_C, actual_rates in dalpha_dt_vals.items():
        T = temp_C + 273.15
        for i, alpha in enumerate(alpha_vals):
            predicted = da_dt(A2, E2, m, n, alpha, T)
            error += (predicted - actual_rates[i]) ** 2
    return error


def find_dalpha_dt_at_alphas(fraction_cured_lst, dalpha_dt_lst, alphas):
    return [np.interp(alpha, fraction_cured_lst, dalpha_dt_lst) for alpha in alphas]


alpha_vals = np.linspace(0.2, 0.95, 500)

dalpha_dt_vals = {
    120: find_dalpha_dt_at_alphas(fraction_cured_120, dalpha_dt_120, alpha_vals),
    150: find_dalpha_dt_at_alphas(fraction_cured_150, dalpha_dt_150, alpha_vals),
    180: find_dalpha_dt_at_alphas(fraction_cured_180, dalpha_dt_180, alpha_vals)
}

# Initial guesses (physical)
log_A2_init = 6.0  # A2 = 1e6
E2_init = 6e4
m_init = 1.2
n_init = 1.25

# Convert to normalized space
x0_norm = params_to_normalized(log_A2_init, E2_init, m_init, n_init)

result = minimize(
    calculate_error_normalized,
    x0=x0_norm,
    args=(dalpha_dt_vals, alpha_vals),
    method='L-BFGS-B',
    bounds=[(0, 1), (0, 1), (0, 1), (0, 1)]
)

A2_solution, E2_solution, m_solution, n_solution = normalized_to_params(result.x)
if __name__ == "__main__":
    print("Optimized parameters:")
    print(f"A2: {A2_solution:.3e}")
    print(f"E2: {E2_solution:.0f}")
    print(f"m: {m_solution:.3f}")
    print(f"n: {n_solution:.3f}")
    print(f"Final error: {result.fun:.6e}")
