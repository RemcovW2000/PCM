import numpy as np
from q1 import final_data_120, final_data_150, final_data_180
from q5 import A1, fraction_cured_120, fraction_cured_150, fraction_cured_180, \
    dalpha_dt_120, dalpha_dt_150, dalpha_dt_180, E1


def da_dt(A2, E2, m, n, alpha, T):
    R = 8.314  # J/(mol·K)
    term_1 = A1 * np.exp(-E1 / (R * T))
    term_2 = A2 * np.exp(-E2 / (R * T)) * alpha ** m
    return (term_1 + term_2) * (1 - alpha)**n

def calculate_error(
        A2:float,
        E2: float,
        m: float,
        n: float,
        dalpha_dt_vals: dict[float, list[float]],
        alpha_vals: list[float]
):
    """
    Calculate the sum of squared errors between predicted and actual dalpha/dt values.

    dalpha_dt_vals: dict mapping the temperature (in celsius) to the list of dalpha/dt values at that temperature from the data.
    alpha_vals: list of alpha values at which to evaluate the dalpha/dt.
    """
    error = 0.0
    for key, lst in dalpha_dt_vals.items():
        T = key + 273.15  # Convert to Kelvin
        for i, alpha in enumerate(alpha_vals):
            predicted = da_dt(A2, E2, m, n, alpha, T)
            actual = dalpha_dt_vals[key][i]
            error += (predicted - actual) ** 2
    return error

def find_dalpha_dt_at_alphas(fraction_cured_lst:list[float], dalpha_dt_lst: list[float], alphas: list[float]) -> list[float]:
    """"""
    dalpha_dt_vals = []
    for alpha in alphas:
        dalpha_dt_vals.append(np.interp(alpha, fraction_cured_lst, dalpha_dt_lst))
    return dalpha_dt_vals

alpha_vals = np.linspace(0.01, 0.97, 1000)
dalpha_dt_vals = {}
dalpha_dt_vals[120] = find_dalpha_dt_at_alphas(
    fraction_cured_lst=fraction_cured_120,
    dalpha_dt_lst=dalpha_dt_120,
    alphas=alpha_vals
)

dalpha_dt_vals[150] = find_dalpha_dt_at_alphas(
    fraction_cured_lst=fraction_cured_150,
    dalpha_dt_lst=dalpha_dt_150,
    alphas=alpha_vals
)

dalpha_dt_vals[180] = find_dalpha_dt_at_alphas(
    fraction_cured_lst=fraction_cured_180,
    dalpha_dt_lst=dalpha_dt_180,
    alphas=alpha_vals
)

if __name__ == "__main__":
    from scipy.optimize import minimize

    # initial guesses
    A2_init = 1e6
    E2_init = 6e4  # J/mol
    m_init = 1
    n_init = 1.25

    result = minimize(
        lambda params: calculate_error(
            A2=params[0],
            E2=params[1],
            m=params[2],
            n=params[3],
            dalpha_dt_vals=dalpha_dt_vals,
            alpha_vals=alpha_vals
        ),
        x0=[A2_init, E2_init, m_init, n_init],
        bounds=[(1e3, 1e10), (1000.0, 200000.0), (0.0, 2.4), (0.0, 2.5)]
    )

    A2_opt, E2_opt, m_opt, n_opt = result.x
    print("Optimized parameters:")
    print("A2:", A2_opt)
    print("E2:", E2_opt)
    print("m:", m_opt)
    print("n:", n_opt)