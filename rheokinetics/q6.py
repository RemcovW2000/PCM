from q5 import DMA_results_by_freq
from rheokinetics.q3 import headers

#---------------------------------------------------------------------------------------
# find max value in loss modulus data:
#---------------------------------------------------------------------------------------

for freq, dataset in DMA_results_by_freq.items():
    log_loss_modulus = dataset[headers.FILTERED_LOG_LOSS_MODULUS.value]

    max_val = max(log_loss_modulus)
    max_index = log_loss_modulus.index(max_val)
    temperature = dataset[headers.TEMP.value][max_index]
    print("Tg at {} Hz: {:.2f} °C".format(freq, temperature))
    dataset['Tg_from_loss_modulus'] = temperature

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    freqs = []
    tgs_intersection_pts = []
    tgs_inflection_pts = []
    tgs_loss_modulus_pts = []
    for freq, dataset in DMA_results_by_freq.items():
        freqs.append(freq)
        tgs_loss_modulus_pts.append(dataset['Tg_from_loss_modulus'])
        tgs_inflection_pts.append(dataset['Tg_from_inflection'])
        tgs_intersection_pts.append(dataset['Tg_from_tangent'])

    plt.plot(freqs, tgs_loss_modulus_pts, label="Tg from Loss Modulus", marker='o')
    plt.plot(freqs, tgs_inflection_pts, label="Tg from Inflection Point", marker='x')
    plt.plot(freqs, tgs_intersection_pts, label="Tg from Tangent Intersection", marker='s')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Tg (°C)')
    plt.title('Tg vs Frequency from Different Methods')
    plt.legend()
    plt.show()