import numpy as np

from q1 import final_data_120, final_data_150, final_data_180

import matplotlib.pyplot as plt

def integrate_heat_flow_rate(heat_flow_rate, time):
    """
    Integrate heat flow rate (W/g) over time (s) to get total heat released (J/g).

    We use the trapezoidal rule for numerical integration.
    """
    total_heat_released = [0.0]
    for i in range(1, len(time)):
        dt = (time[i] - time[i - 1])
        avg_heat_flow = (heat_flow_rate[i] + heat_flow_rate[i - 1]) / 2
        total_heat_released.append(total_heat_released[-1] + avg_heat_flow * dt)
    return total_heat_released

heat_flow_120 = integrate_heat_flow_rate(final_data_120['Filtered Heat Flow'], final_data_120['Time Seconds'])
heat_flow_150 = integrate_heat_flow_rate(final_data_150['Filtered Heat Flow'], final_data_150['Time Seconds'])
heat_flow_180 = integrate_heat_flow_rate(final_data_180['Filtered Heat Flow'], final_data_180['Time Seconds'])

delta_H_120 = heat_flow_120[-1]
delta_H_150 = heat_flow_150[-1]
delta_H_180 = heat_flow_180[-1]

delta_H_max = max(delta_H_120, delta_H_150, delta_H_180)

fraction_cured_120 = [h / delta_H_max for h in heat_flow_120]
cure_rate_120 = np.gradient(fraction_cured_120, final_data_120['Time Seconds'])

fraction_cured_150 = [h / delta_H_max for h in heat_flow_150]
cure_rate_150 = np.gradient(fraction_cured_150, final_data_150['Time Seconds'])

fraction_cured_180 = [h / delta_H_max for h in heat_flow_180]
cure_rate_180 = np.gradient(fraction_cured_180, final_data_180['Time Seconds'])

if __name__ == "__main__":
    """Plot degree of cure vs. time for the three isothermal datasets."""
    plt.plot(final_data_120['Time'], fraction_cured_120, label='120°C')
    plt.plot(final_data_150['Time'], fraction_cured_150, label='150°C')
    plt.plot(final_data_180['Time'], fraction_cured_180, label='180°C')
    plt.legend()
    plt.xlabel('Time (minutes)')
    plt.ylabel(r'Degree of cure, $\alpha$')
    plt.title('Degree of cure vs. Time, ')
    plt.xlim(0, 140)
    plt.ylim(0, 1.05)
    plt.axhline(y=1.0, color='black', linestyle='--')
    plt.show()

    """log plot for comparison:"""
    plt.plot(final_data_120['Time'], fraction_cured_120, label='120°C')
    plt.plot(final_data_150['Time'], fraction_cured_150, label='150°C')
    plt.plot(final_data_180['Time'], fraction_cured_180, label='180°C')
    plt.legend()
    plt.xlabel('Time (minutes)')
    plt.ylabel(r'Degree of cure, $\alpha$')
    plt.title('Degree of cure vs. Time')
    plt.xlim(1, 10e3)
    plt.xscale('log')
    plt.ylim(0, 1.1)
    plt.axhline(y=1.0, color='black', linestyle='--')
    plt.show()