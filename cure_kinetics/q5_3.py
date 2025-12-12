import numpy as np
from q3 import fraction_cured_150, cure_rate_150
from q5 import E1_solution, A1_solution
from q5_2 import m_solution, n_solution, E2_solution, A2_solution, simulate_cure

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    t_lst = np.linspace(0, 30000, 10000)

    cure_temp = 150
    sim_results = simulate_cure(A1_solution, E1_solution, A2_solution, E2_solution, m_solution, n_solution, cure_temp, t_lst)
    alpha_vals = sim_results[0]
    da_dt_vals = sim_results[1]
    plt.plot(alpha_vals, da_dt_vals, label=f'Simulated {cure_temp}°C')

    plt.plot(fraction_cured_150, cure_rate_150, '--', label='Experimental 150°C')
    plt.xlabel('Time (s)')
    plt.ylabel('Degree of Cure (α)')
    plt.title('Cure rate vs. degree of cure, simulated vs. experimental')
    plt.axhline(0, color='k', linestyle='--')
    plt.legend()
    plt.show()