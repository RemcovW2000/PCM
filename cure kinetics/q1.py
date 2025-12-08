import numpy as np
from multiprocessing import Value
from pathlib import Path
from typing import Dict, List
import re

from matplotlib import pyplot as plt
from resources.constants import START_TIME_120, START_TIME_150, SAMPLE_WEIGHT_120, \
    SAMPLE_WEIGHT_150, SAMPLE_WEIGHT_180
from scipy.signal import butter, filtfilt


def read_sheet_to_dict(path: Path) -> Dict[str, List[Value]]:
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

    result: Dict[str, List[Value]] = {h: [] for h in header}
    for row in rows:
        # if a row is shorter/longer, zip will clip to header length
        for h, cell in zip(header, row):
            s = cell.replace(',', '.')
            value = float(s)
            result[h].append(value)
    return result

def apply_lowpass_filter(data: list[float], time: list[float], cutoff_freq: float) -> np.ndarray:
    t = np.array(time)
    x = np.array(data)

    # 1) sampling frequency (assumes ~uniform spacing)
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt

    # 2) design low-pass filter
    order = 2
    b, a = butter(order, cutoff_freq / (0.5 * fs), btype="low")

    # 3) apply zero-phase filter
    x_lp = filtfilt(b, a, x)
    return x_lp.tolist()

def prepare_for_plotting(data: Dict[str, list[float]], sample_weight: float, start_time: float = 0.0) ->Dict[str, list[float]]:
    """
    Prepare the data for plotting by filtering and normalizing.
    """
    time = np.array(data['Time'])
    unsubtracted_heat_flow = np.array(data['Unsubtracted'])
    baseline_heat_flow = np.array(data['Baseline'])

    # Remove numerical spikes (as seen in 180 deg data)
    for i, flow in enumerate(unsubtracted_heat_flow[:-1]):
        if flow / unsubtracted_heat_flow[i+1] >5:
            unsubtracted_heat_flow[i+1] = flow

    # subtract baseline (which is 0 in the dataset)
    net_heat_flow = unsubtracted_heat_flow - baseline_heat_flow

    # Convert to W/g
    net_heat_flow = net_heat_flow / sample_weight

    # Normalize to 0 mW at the end of the measurement
    final_heat_flow_at_end = net_heat_flow[-100:].mean()
    net_heat_flow -= final_heat_flow_at_end

    # Find the first exothermic event (net heat flow > 0):
    index_first_exotherm = next(
        i for i, v in enumerate(net_heat_flow) if v > 0)

    # Find the index corresponding to the specified start time
    index_at_start_time = next(
        i for i, t in enumerate(time) if t >= start_time)

    first_index = max(index_first_exotherm, index_at_start_time)

    # copy data from the given data dict.
    data_out = {}
    for key, item in data.items():
        data_out[key] = item[first_index:]

    # add net heat flow
    data_out['Net Heat Flow'] = net_heat_flow[first_index:].tolist()

    # apply low-pass filter to heat flow
    data_out['Filtered Heat Flow'] = apply_lowpass_filter(
        data_out['Net Heat Flow'], data_out['Time'], cutoff_freq=1)
    return data_out


isothermal_120_path = Path(__file__).parent / "resources/isothermal_120.txt"
data_120 = read_sheet_to_dict(isothermal_120_path)

final_data_120 = prepare_for_plotting(data_120, sample_weight= SAMPLE_WEIGHT_120,start_time=START_TIME_120)
time_120 = final_data_120['Time']
net_heat_flow_120 = final_data_120['Filtered Heat Flow']


isothermal_150_path = Path(__file__).parent / "resources/isothermal_150.txt"
data_150 = read_sheet_to_dict(isothermal_150_path)

final_data_150 = prepare_for_plotting(data_150, sample_weight= SAMPLE_WEIGHT_150,start_time=START_TIME_150)
time_150 = final_data_150['Time']
net_heat_flow_150 = final_data_150['Filtered Heat Flow']

isothermal_180_path = Path(__file__).parent / "resources/isothermal_180.txt"
data_180 = read_sheet_to_dict(isothermal_180_path)
final_data_180 = prepare_for_plotting(data_180, sample_weight=SAMPLE_WEIGHT_180)
time_180 = final_data_180['Time']
net_heat_flow_180 = final_data_180['Filtered Heat Flow']

plt.plot(time_120, net_heat_flow_120, label='120°C')
plt.plot(time_150, net_heat_flow_150, label='150°C')
plt.plot(time_180, net_heat_flow_180, label='180°C')
plt.xlim(0, 200)
plt.axhline(y=0, color='black', linestyle='--')
plt.axvline(x=0, color='black', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Net specific heat flow (W/g)')
plt.title('Net specific heat flow vs. Time')
plt.legend()
plt.show()
