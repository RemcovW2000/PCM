import re
from pathlib import Path
from typing import Dict, List

from matplotlib import pyplot as plt


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

path_to_sheet = Path(__file__).parent / "resources" / "DMA_results.txt"
DMA_results = read_sheet_to_dict(path_to_sheet)

#---------------------------------------------------------------------------------------
# Remove 'appendix' of data where temperature goes back down:
#---------------------------------------------------------------------------------------
max_temp_index = DMA_results["Temp."].index(max(DMA_results["Temp."]))
for key in DMA_results.keys():
    DMA_results[key] = DMA_results[key][:max_temp_index + 1]

DMA_results_20_hz = filter_dict_by_value(DMA_results, key="Freq.", upper_limit=20.0, lower_limit=20.0)
DMA_results_10_hz = filter_dict_by_value(DMA_results, key="Freq.", upper_limit=10.0, lower_limit=10.0)
DMA_results_5_hz = filter_dict_by_value(DMA_results, key="Freq.", upper_limit=5.0, lower_limit=5.0)
DMA_results_1_hz = filter_dict_by_value(DMA_results, key="Freq.", upper_limit=1.0, lower_limit=1.0)
DMA_results_02_hz = filter_dict_by_value(DMA_results, key="Freq.", upper_limit=0.2, lower_limit=0.2)

if __name__ == "__main__":
    plt.plot(DMA_results_20_hz["Temp."], DMA_results_20_hz["E'(G')"], label='20 Hz')
    plt.plot(DMA_results_10_hz["Temp."], DMA_results_10_hz["E'(G')"], label='10 Hz')
    plt.plot(DMA_results_5_hz["Temp."], DMA_results_5_hz["E'(G')"], label='5 Hz')
    plt.plot(DMA_results_1_hz["Temp."], DMA_results_1_hz["E'(G')"], label='1 Hz')
    plt.plot(DMA_results_02_hz["Temp."], DMA_results_02_hz["E'(G')"], label='0.2 Hz')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Temperature (°C)')
    plt.ylabel("Storage Modulus E' (Pa)")
    plt.title("Storage Modulus vs Temperature at Different Frequencies")
    plt.show()

    plt.plot(DMA_results_20_hz["Temp."], DMA_results_20_hz['E"(G")'], label='20 Hz')
    plt.plot(DMA_results_10_hz["Temp."], DMA_results_10_hz['E"(G")'], label='10 Hz')
    plt.plot(DMA_results_5_hz["Temp."], DMA_results_5_hz['E"(G")'], label='5 Hz')
    plt.plot(DMA_results_1_hz["Temp."], DMA_results_1_hz['E"(G")'], label='1 Hz')
    plt.plot(DMA_results_02_hz["Temp."], DMA_results_02_hz['E"(G")'], label='0.2 Hz')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Loss Modulus E"(G") (Pa)')
    plt.title("Loss Modulus vs Temperature at Different Frequencies")
    plt.show()