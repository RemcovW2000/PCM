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

    log_E_prime = dataset[headers.LOG_STORAGE_MODULUS.value]
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

    def intersect(self, other: 'straight_line') -> Point2D:
        if self.slope == other.slope:
            raise ValueError("Lines are parallel and do not intersect.")

        x_intersect = (other.intersection_point.y - self.intersection_point.y +
                       self.slope * self.intersection_point.x -
                       other.slope * other.intersection_point.x) / (self.slope - other.slope)

        y_intersect = self.value_at(x_intersect)
        return Point2D(x_intersect, y_intersect)

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

# ---------------------------------------------------------------------------------------
# find 2nd derivative, find inflection points where 2nd derivative crosses zero
# ---------------------------------------------------------------------------------------

for freq, dataset in DMA_results_by_freq.items():
    log_E_d_prime = dataset[headers.DERIVATIVE_LOG_STORAGE_MODULUS.value]
    temperature = dataset[headers.TEMP.value]
    dlogE_dT = np.gradient(log_E_d_prime, temperature)
    dataset[headers.DERIVATIVE_2_LOG_STORAGE_MODULUS.value] = dlogE_dT

    log_E_d_prime = dataset[headers.DERIVATIVE_LOG_LOSS_MODULUS.value]
    dlogE_dT = np.gradient(log_E_d_prime, temperature)
    dataset[headers.DERIVATIVE_2_LOG_STORAGE_MODULUS.value] = dlogE_dT

    start_index = next(
        i for i, temp in enumerate(dataset[headers.TEMP.value]) if temp >= 85.0)
    end_index = next(
        i for i, temp in enumerate(dataset[headers.TEMP.value]) if temp >= 105.0)

    second_derivative_segment = dataset[headers.DERIVATIVE_2_LOG_STORAGE_MODULUS.value][
                                start_index:end_index]
    temp_segment = dataset[headers.TEMP.value][start_index:end_index]
    zero_crossings = np.where(np.diff(np.sign(second_derivative_segment)))[0]
    inflection_points = []
    for crossing in zero_crossings:
        t1 = temp_segment[crossing]
        t2 = temp_segment[crossing + 1]
        y1 = second_derivative_segment[crossing]
        y2 = second_derivative_segment[crossing + 1]
        t_inflect = t1 - y1 * (t2 - t1) / (y2 - y1)
        inflection_points.append(t_inflect)

    if len(inflection_points) > 1:
        raise ValueError("Multiple inflection points found in the specified range.")

    inflection_pt = inflection_points[0]
    dataset['inflection_point'] = inflection_pt


fig, ax = plt.subplots(5, 1, sharex=True, figsize=(7, 10))

for i, freq in enumerate(sorted(DMA_results_by_freq.keys())):
    ax[i].plot(
        DMA_results_by_freq[freq]["Temp."],
        DMA_results_by_freq[freq][headers.FILTERED_LOG_STORAGE_MODULUS.value],
        label=f'{freq} Hz'
    )
    ax[i].set_ylim(7.5, 10)
    ax[i].set_xlim(70, 120)

    ax[i].axline(
        xy1=(
            DMA_results_by_freq[freq]['tangent_line_at_min'].intersection_point.x,
            DMA_results_by_freq[freq]['tangent_line_at_min'].intersection_point.y
        ),
        slope=DMA_results_by_freq[freq]['tangent_line_at_min'].slope,
        color='orange',
        linestyle='--',
    )

    ax[i].axline(
        xy1=(
            DMA_results_by_freq[freq]['tangent_line_at_start'].intersection_point.x,
            DMA_results_by_freq[freq]['tangent_line_at_start'].intersection_point.y
        ),
        slope=DMA_results_by_freq[freq]['tangent_line_at_start'].slope,
        color='orange',
        linestyle='--',
    )
    ax[i].legend(loc='upper right', fontsize='small')

    intersection_pt = DMA_results_by_freq[freq]['tangent_line_at_min'].intersect(
        DMA_results_by_freq[freq]['tangent_line_at_start'])


    tg_from_tangent_lines = intersection_pt.x

    ax[i].axvline(
        x=intersection_pt.x,
        color='red',
        linestyle=':'
    )

    label = f"Tg (intersection) = {tg_from_tangent_lines:.2f} °C"  # format as desired

    ax[i].annotate(
        label,
        xy=(tg_from_tangent_lines, 1.0),
        xycoords=('data', 'axes fraction'),
        # x in data coords, y as fraction of the axes height
        xytext=(0, 6),  # nudge label a few points above the top
        textcoords='offset points',
        ha='right',
        va='bottom',
        color='red',
        fontsize='small',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8)
    )

    tg_from_inflection_point = DMA_results_by_freq[freq]['inflection_point']
    ax[i].axvline(
        x=tg_from_inflection_point,
        color='red',
        linestyle=':'
    )
    label = f"Tg (inflection) = {tg_from_inflection_point:.2f} °C"

    ax[i].annotate(
        label,
        xy=(tg_from_inflection_point, 1.0),
        xycoords=('data', 'axes fraction'),
        xytext=(0, 6),
        textcoords='offset points',
        ha='left',
        va='bottom',
        color='red',
        fontsize='small',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8)
    )
    DMA_results_by_freq[freq]['Tg_from_tangent'] = tg_from_tangent_lines
    DMA_results_by_freq[freq]['Tg_from_inflection'] = tg_from_inflection_point

# Figure-level labels and title
fig.suptitle(
    "Tg determined from Tangent Line Intersection and Inflection Point Methods",)
fig.supxlabel('Temperature (°C)')
fig.supylabel("dlogE'/dT (1/°C)")

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
