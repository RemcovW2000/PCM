import enum
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

class headers(enum.Enum):
    TEMP = "Temp."
    FREQ = "Freq."
    STORAGE_MODULUS = "E'(G')"
    LOSS_MODULUS = 'E"(G")'

    LOG_STORAGE_MODULUS = "log_E'"
    LOG_LOSS_MODULUS = 'log_E"'
    FILTERED_LOG_STORAGE_MODULUS = "log_E'_lp"
    FILTERED_LOG_LOSS_MODULUS = 'log_E"_lp'

    DERIVATIVE_LOG_STORAGE_MODULUS = "dlogE'_dT"
    DERIVATIVE_LOG_LOSS_MODULUS = 'dlogE"_dT'

    DERIVATIVE_2_LOG_STORAGE_MODULUS = "d2logE'_dT2"
    DERIVATIVE_2_LOG_LOSS_MODULUS = 'd2logE"_dT2'


def read_sheet_to_dict(path: Path) -> Dict[str, List[float]]:
    """
    Read a whitespace/tab-separated table from `path` and return a dict
    mapping header -> list of column values. Decimal commas (`,`) are
    accepted and converted to dots for numeric parsing.
    """
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    # split header on any whitespace
    header = re.split(r'\s+', lines[0])
    rows: List[List[str]] = []
    for ln in lines[1:]:
        parts = re.split(r'\s+', ln)
        rows.append(parts)

    result: Dict[str, List[float]] = {h: [] for h in header}
    for row in rows:
        # if a row is shorter/longer, zip will clip to header length
        for h, cell in zip(header, row):
            s = cell.replace(',', '.')
            value = float(s)
            result[h].append(value)
    return result

def filter_dict_by_value(data: Dict[str, List[float]], key: str, upper_limit: float, lower_limit: float) -> Dict[str, List[float]]:
    """
    Filter the input dictionary `data` to only include rows where the value
    in the column `key` is between upper and lower limit.
    """
    if key not in data:
        raise KeyError(f"Key '{key}' not found in data.")

    filtered_indices = [i for i, value in enumerate(data[key]) if upper_limit >= value >= lower_limit]

    filtered_data: Dict[str, List[float]] = {k: [] for k in data.keys()}
    for i in filtered_indices:
        for k in data.keys():
            filtered_data[k].append(data[k][i])

    return filtered_data

def apply_lowpass_filter(data: list[float], time: list[float], cutoff_freq: float) -> np.ndarray:
    t = np.array(time)
    x = np.array(data)

    if len(x) < 9:
        raise ValueError("Data length must be at least 9 to apply the filter.")

    # 1) sampling frequency (assumes ~uniform spacing)
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt

    # 2) design low-pass filter
    order = 2
    b, a = butter(order, cutoff_freq / (0.5 * fs), btype="low")

    # 3) apply zero-phase filter
    x_lp = filtfilt(b, a, x)
    return x_lp.tolist()

path_to_sheet = Path(__file__).parent / "resources" / "DMA_results.txt"
DMA_results = read_sheet_to_dict(path_to_sheet)

#---------------------------------------------------------------------------------------
# Remove 'appendix' of data where temperature goes back down:
#---------------------------------------------------------------------------------------
max_temp_index = DMA_results["Temp."].index(max(DMA_results["Temp."]))
for key in DMA_results.keys():
    DMA_results[key] = DMA_results[key][:max_temp_index + 1]

#---------------------------------------------------------------------------------------
# Organize data by frequency:
#---------------------------------------------------------------------------------------
DMA_results_by_freq: Dict[float, Dict[str, List[float]]] = {
    20.0: filter_dict_by_value(DMA_results, key="Freq.", upper_limit=20.0, lower_limit=20.0),
    10.0: filter_dict_by_value(DMA_results, key="Freq.", upper_limit=10.0, lower_limit=10.0),
    5.0: filter_dict_by_value(DMA_results, key="Freq.", upper_limit=5.0, lower_limit=5.0),
    1.0: filter_dict_by_value(DMA_results, key="Freq.", upper_limit=1.0, lower_limit=1.0),
    0.2: filter_dict_by_value(DMA_results, key="Freq.", upper_limit=0.3, lower_limit=0.1),
}

#---------------------------------------------------------------------------------------
# add log data to datasets:
#---------------------------------------------------------------------------------------
for freq, dataset in DMA_results_by_freq.items():
    dataset[headers.LOG_STORAGE_MODULUS.value] = [np.log10(value) for value in dataset[headers.STORAGE_MODULUS.value]]
    dataset[headers.LOG_LOSS_MODULUS.value] = [np.log10(value) for value in dataset[headers.LOSS_MODULUS.value]]

#---------------------------------------------------------------------------------------
# Apply low-pass filter to log data:
#---------------------------------------------------------------------------------------

for freq, dataset in DMA_results_by_freq.items():
    dataset[headers.FILTERED_LOG_LOSS_MODULUS.value] = apply_lowpass_filter(
        data=dataset[headers.LOG_LOSS_MODULUS.value],
        time=dataset[headers.TEMP.value],
        cutoff_freq=0.25  # 1/°C
    )
    dataset[headers.FILTERED_LOG_STORAGE_MODULUS.value] = apply_lowpass_filter(
        data=dataset[headers.LOG_STORAGE_MODULUS.value],
        time=dataset[headers.TEMP.value],
        cutoff_freq=0.25 # 1/°C
    )

# --------------------------------------------------------------------------------------
# convert filtered log data back to linear scale:
# --------------------------------------------------------------------------------------
for freq, dataset in DMA_results_by_freq.items():
    dataset["E'_lp"] = [10**value for value in dataset["log_E'_lp"]]
    dataset['E"_lp'] = [10**value for value in dataset['log_E"_lp']]

if __name__ == "__main__":
    plt.plot(DMA_results_by_freq[20.0][headers.TEMP.value], DMA_results_by_freq[20.0][headers.LOG_STORAGE_MODULUS.value],
             label='20 Hz')
    plt.plot(DMA_results_by_freq[10.0][headers.TEMP.value], DMA_results_by_freq[10.0][headers.LOG_STORAGE_MODULUS.value],
             label='10 Hz')
    plt.plot(DMA_results_by_freq[5.0][headers.TEMP.value], DMA_results_by_freq[5.0][headers.LOG_STORAGE_MODULUS.value],
             label='5 Hz')
    plt.plot(DMA_results_by_freq[1.0][headers.TEMP.value], DMA_results_by_freq[1.0][headers.LOG_STORAGE_MODULUS.value],
             label='1 Hz')
    plt.plot(DMA_results_by_freq[0.2][headers.TEMP.value], DMA_results_by_freq[0.2][headers.LOG_STORAGE_MODULUS.value],
             label='0.2 Hz')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Temperature (°C)')
    plt.ylabel("Storage Modulus E' (Pa)")
    plt.title("Storage Modulus vs Temperature at Different Frequencies")
    plt.show()

    plt.plot(DMA_results_by_freq[20.0]["Temp."], DMA_results_by_freq[20.0]['log_E"_lp'], label='20 Hz')
    plt.plot(DMA_results_by_freq[10.0]["Temp."], DMA_results_by_freq[10.0]['log_E"_lp'], label='10 Hz')
    plt.plot(DMA_results_by_freq[5.0]["Temp."], DMA_results_by_freq[5.0]['log_E"_lp'], label='5 Hz')
    plt.plot(DMA_results_by_freq[1.0]["Temp."], DMA_results_by_freq[1.0]['log_E"_lp'], label='1 Hz')
    plt.plot(DMA_results_by_freq[0.2]["Temp."], DMA_results_by_freq[0.2]['log_E"_lp'], label='0.2 Hz')
    plt.legend()
    # plt.yscale('log')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Loss Modulus E"(G") (Pa)')
    plt.title("Loss Modulus vs Temperature at Different Frequencies")
    plt.show()