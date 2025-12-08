import numpy as np
from multiprocessing import Value
from pathlib import Path
from typing import Dict, List
import re

from matplotlib import pyplot as plt


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

def prepare_for_plotting(data: Dict[str, list[float]]) ->Dict[str, list[float]]:
    """
    Prepare the data for plotting by filtering and normalizing.
    """
    time = np.array(data['Time'])
    unsubtracted_heat_flow = np.array(data['Unsubtracted'])
    baseline_heat_flow = np.array(data['Baseline'])

    net_heat_flow = unsubtracted_heat_flow - baseline_heat_flow

    index_first_exotherm = next(
        i for i, v in enumerate(net_heat_flow) if v > 0)

    data_out = {}
    for key, item in data.items():
        data_out[key] = item[index_first_exotherm:]

    data_out['Net Heat Flow'] = net_heat_flow[index_first_exotherm:].tolist()
    return data_out


isothermal_120_path = Path(__file__).parent / "resources/isothermal_120.txt"
data_120 = read_sheet_to_dict(isothermal_120_path)

final_data_120 = prepare_for_plotting(data_120)
time_120 = final_data_120['Time']
net_heat_flow_120 = final_data_120['Net Heat Flow']


isothermal_150_path = Path(__file__).parent / "resources/isothermal_150.txt"
data_150 = read_sheet_to_dict(isothermal_150_path)
final_data_150 = prepare_for_plotting(data_150)
time_150 = final_data_150['Time']
net_heat_flow_150 = final_data_150['Net Heat Flow']

isothermal_180_path = Path(__file__).parent / "resources/isothermal_180.txt"
data_180 = read_sheet_to_dict(isothermal_180_path)
final_data_180 = prepare_for_plotting(data_180)
time_180 = final_data_180['Time']
net_heat_flow_180 = final_data_180['Net Heat Flow']

plt.plot(time_120, net_heat_flow_120, label='120°C')
plt.plot(time_150, net_heat_flow_150, label='150°C')
plt.plot(time_180, net_heat_flow_180, label='180°C')
plt.axhline(y=0, color='black', linestyle='--')
plt.axvline(x=0, color='black', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Net Heat Flow (mW)')
plt.legend()
plt.show()
