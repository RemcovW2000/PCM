import numpy as np

from q1 import final_data_150, final_data_120, final_data_180
from q3 import fraction_cured_150, fraction_cured_120, fraction_cured_180
from q5 import E1_solution, A1_solution
from q5_2 import m_solution, n_solution, E2_solution, A2_solution


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
    return np.array(alpha_vals), np.array(da_dt_vals)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # Example parameters
    t_lst = np.linspace(0, 30000, 1000000)

    for cure_temp in [120, 150, 180]:
        sim_results = simulate_cure(A1_solution, E1_solution, A2_solution, E2_solution, m_solution, n_solution, cure_temp, t_lst)[0]
        plt.plot(t_lst, sim_results, label=f'Simulated {cure_temp}°C')

    plt.plot(final_data_120["Time Seconds"], fraction_cured_120, '--')
    plt.plot(final_data_150["Time Seconds"], fraction_cured_150, '--')
    plt.plot(final_data_180["Time Seconds"], fraction_cured_180, '--')
    plt.xlim(0, 10000)
    plt.show()