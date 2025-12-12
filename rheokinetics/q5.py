# --------------------------------------------------------------------------------------
# Add log data to datasets:
# --------------------------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt

from rheokinetics.q3 import DMA_results_by_freq, headers

# find derivative of log_E' with respect to Temp.

for freq, dataset in DMA_results_by_freq.items():
    log_E_prime = dataset[headers.LOG_STORAGE_MODULUS.value]
    temperature = dataset[headers.TEMP.value]
    dlogE_dT = np.gradient(log_E_prime, temperature)
    dataset[headers.DERIVATIVE_LOG_STORAGE_MODULUS.value] = dlogE_dT

    log_E_prime = dataset[headers.LOG_LOSS_MODULUS.value]
    dlogE_dT = np.gradient(log_E_prime, temperature)
    dataset[headers.DERIVATIVE_LOG_LOSS_MODULUS.value] = dlogE_dT

# --------------------------------------------------------------------------------------
# find index at which derivative is maximum
# --------------------------------------------------------------------------------------

class Point2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class straight_line:
    def __init__(self, slope: float, intersection_point: Point2D):
        self.slope = slope
        self.intersection_point = intersection_point

    def value_at(self, x: float) -> float:
        return self.slope * (x - self.intersection_point.x) + self.intersection_point.y

for freq, dataset in DMA_results_by_freq.items():
    index_at_120_deg = next(i for i, temp in enumerate(dataset[headers.TEMP.value]) if temp >= 120.0)
    dlogE_dT = dataset[headers.DERIVATIVE_LOG_STORAGE_MODULUS.value][:index_at_120_deg]
    min_index = np.argmin(dlogE_dT)

    slope = dlogE_dT[min_index]
    intersection_point = Point2D(
        x=dataset[headers.TEMP.value][min_index],
        y=dataset[headers.FILTERED_LOG_STORAGE_MODULUS.value][min_index]
    )
    dataset['tangent_line_at_min'] = straight_line(slope, intersection_point)

    index_at_0_deg = next(
        i for i, temp in enumerate(dataset[headers.TEMP.value]) if temp >= 0.0)

    index_at_60_deg = next(
        i for i, temp in enumerate(dataset[headers.TEMP.value]) if temp >= 60.0)

    slope_start = np.mean(dlogE_dT[index_at_0_deg:index_at_60_deg])
    intersection_point = Point2D(
        x=dataset[headers.TEMP.value][index_at_0_deg],
        y=dataset[headers.FILTERED_LOG_STORAGE_MODULUS.value][index_at_0_deg]
    )
    dataset['tangent_line_at_start'] = straight_line(slope_start, intersection_point)

if __name__ == "__main__":
    fig, ax = plt.subplots()

    ax.plot(DMA_results_by_freq[20.0]["Temp."], DMA_results_by_freq[20.0][headers.FILTERED_LOG_STORAGE_MODULUS.value],
             label='20 Hz')
    ax.set_ylim(7.5, 10)
    ax.axline(
        xy1=(
            DMA_results_by_freq[20.0]['tangent_line_at_min'].intersection_point.x,
            DMA_results_by_freq[20.0]['tangent_line_at_min'].intersection_point.y
        ),
        slope=DMA_results_by_freq[20.0]['tangent_line_at_min'].slope,
        color='orange',
        linestyle='--',
        label='Tangent at min dlogE\'/dT (20 Hz)'
    )

    ax.axline(
        xy1=(
            DMA_results_by_freq[20.0]['tangent_line_at_start'].intersection_point.x,
            DMA_results_by_freq[20.0]['tangent_line_at_start'].intersection_point.y
        ),
        slope=DMA_results_by_freq[20.0]['tangent_line_at_start'].slope,
        color='orange',
        linestyle='--',
        label='Tangent at min dlogE\'/dT (20 Hz)'
    )

    plt.legend()
    plt.xlabel('Temperature (°C)')
    plt.ylabel("dlogE'/dT (1/°C)")
    plt.title("Derivative of log Storage Modulus vs Temperature at Different Frequencies")
    plt.show()