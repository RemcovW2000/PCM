# --------------------------------------------------------------------------------------
# Add log data to datasets:
# --------------------------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt

from rheokinetics.q3 import DMA_results_by_freq

# find derivative of log_E' with respect to Temp.

for freq, dataset in DMA_results_by_freq.items():
    log_E_prime = dataset["log_E'_lp"]
    temperature = dataset["Temp."]
    dlogE_dT = np.gradient(log_E_prime, temperature)
    dataset["dlogE'_dT"] = dlogE_dT

    log_E_prime = dataset['log_E"_lp']
    dlogE_dT = np.gradient(log_E_prime, temperature)
    dataset['dlogE"_dT'] = dlogE_dT

if __name__ == "__main__":
    plt.plot(DMA_results_by_freq[20.0]["Temp."], DMA_results_by_freq[20.0]['dlogE"_dT'],
             label='20 Hz')
    plt.plot(DMA_results_by_freq[10.0]["Temp."], DMA_results_by_freq[10.0]['dlogE"_dT'],
             label='10 Hz')
    plt.plot(DMA_results_by_freq[5.0]["Temp."], DMA_results_by_freq[5.0]['dlogE"_dT'],
             label='5 Hz')
    plt.plot(DMA_results_by_freq[1.0]["Temp."], DMA_results_by_freq[1.0]['dlogE"_dT'],
             label='1 Hz')
    plt.plot(DMA_results_by_freq[0.2]["Temp."], DMA_results_by_freq[0.2]['dlogE"_dT'],
             label='0.2 Hz')
    plt.legend()
    plt.xlabel('Temperature (°C)')
    plt.ylabel("dlogE'/dT (1/°C)")
    plt.title("Derivative of log Storage Modulus vs Temperature at Different Frequencies")
    plt.show()