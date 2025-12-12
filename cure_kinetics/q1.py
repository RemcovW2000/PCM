import numpy as np
from multiprocessing import Value
from pathlib import Path
from typing import Dict, List, Optional
import re

from matplotlib import pyplot as plt
from resources.constants import START_TIME_120, START_TIME_150, SAMPLE_WEIGHT_120, \
    SAMPLE_WEIGHT_150, SAMPLE_WEIGHT_180, RAW_DATA_CUTOFF_FREQ, \
    FRACTION_OF_DATA_TO_AVERAGE_FOR_BASELINE, END_TIME_120, TIME_UNIT_CONVERSION_FACTOR
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

def window_data(data: Dict[str, list[float]], start_time: Optional[float] = None, end_time: Optional[float]=None) -> Dict[str, list[float]]:
    """
    Window the data between start_time (or first exothermic event) and end_time.
    """
    time = np.array(data['Time'])

    if start_time:
        index_at_start_time = next(
            i for i, t in enumerate(time) if t >= start_time)
    else:
        index_at_start_time = 0

    if end_time:
        index_at_end_time = next(
            i for i, t in enumerate(time) if t >= end_time)
    else:
        index_at_end_time = len(time)

    unsubtracted_heat_flow = np.array(data['Unsubtracted'])
    baseline_heat_flow = np.array(data['Baseline'])
    net_heat_flow = unsubtracted_heat_flow - baseline_heat_flow

    index_first_exotherm = next(
        i for i, v in enumerate(net_heat_flow) if v > 0)

    start_index = max(index_first_exotherm, index_at_start_time)

    # copy data from the given data dict.
    data_out = {}
    for key, item in data.items():
        data_out[key] = item[start_index:index_at_end_time]
    return data_out

def prepare_for_plotting(data: Dict[str, list[float]], sample_weight: float, start_time: Optional[float]=None, end_time: Optional[float]= None) ->Dict[str, list[float]]:
    """
    Prepare the data for plotting by filtering and normalizing.
    """
    windowed_data = window_data(data, start_time, end_time)

    unsubtracted_heat_flow = np.array(windowed_data['Unsubtracted'])
    baseline_heat_flow = np.array(windowed_data['Baseline'])
    net_heat_flow = unsubtracted_heat_flow - baseline_heat_flow

    # Convert to W/g
    net_heat_flow = net_heat_flow / sample_weight

    # Remove numerical spikes (as seen in 180 deg data)
    for i, flow in enumerate(unsubtracted_heat_flow[:-2]):
        if flow / unsubtracted_heat_flow[i+1] >5:
            unsubtracted_heat_flow[i+1] = (flow + unsubtracted_heat_flow[i+2]) / 2

    # Normalize to 0 mW at the end of the measurement
    nr_indices = FRACTION_OF_DATA_TO_AVERAGE_FOR_BASELINE * len(net_heat_flow)
    final_heat_flow_at_end = net_heat_flow[-int(nr_indices):].mean()
    net_heat_flow -= final_heat_flow_at_end

    # copy data from the given data dict.
    data_out = {}
    for key, item in windowed_data.items():
        data_out[key] = item

    # add net heat flow
    data_out['Net Heat Flow'] = net_heat_flow.tolist()

    # apply low-pass filter to heat flow
    data_out['Filtered Heat Flow'] = apply_lowpass_filter(
        data_out['Net Heat Flow'], data_out['Time'], cutoff_freq=RAW_DATA_CUTOFF_FREQ)

    data_out['Time Seconds'] = [t * TIME_UNIT_CONVERSION_FACTOR for t in data_out['Time']]
    return data_out


isothermal_120_path = Path(__file__).parent / "resources/isothermal_120.txt"
data_120 = read_sheet_to_dict(isothermal_120_path)

final_data_120 = prepare_for_plotting(data_120, sample_weight= SAMPLE_WEIGHT_120,start_time=START_TIME_120, end_time=END_TIME_120)
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

if __name__ == "__main__":
    plt.plot(time_120, net_heat_flow_120, label='120°C')
    plt.plot(time_150, net_heat_flow_150, label='150°C')
    plt.plot(time_180, net_heat_flow_180, label='180°C')
    plt.xlim(0, 120)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Net specific heat flow rate (W/g)')
    plt.title('Net specific heat flow vs. Time')
    plt.legend()
    plt.show()