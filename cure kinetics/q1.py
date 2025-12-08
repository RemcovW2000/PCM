import numpy as np
from scipy.signal import butter, filtfilt
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

isothermal_120_path = Path(__file__).parent / "resources/isothermal_120.txt"
isothermal_150_path = Path(__file__).parent / "resources/isothermal_150.txt"
isothermal_180_path = Path(__file__).parent / "resources/isothermal_180.txt"

data_120 = read_sheet_to_dict(isothermal_120_path)
data_150 = read_sheet_to_dict(isothermal_120_path)
data_180 = read_sheet_to_dict(isothermal_120_path)


time = data_120['Time']
unsubtracted_heat_flow = data_120['Unsubtracted']

t = np.array(time)
x = np.array(unsubtracted_heat_flow)

# 1) sampling frequency (assumes ~uniform spacing)
dt = t[1] - t[0]
fs = 1.0 / dt

# 2) design low-pass filter
cutoff_hz = 0.000001   # choose this based on what you want to keep
order = 2
b, a = butter(order, cutoff_hz / (0.5 * fs), btype="low")

# 3) apply zero-phase filter
x_lp = filtfilt(b, a, x)

plt.plot(time, x_lp, label='120°C')
plt.show()
