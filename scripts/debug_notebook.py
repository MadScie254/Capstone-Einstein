
import json
from pathlib import Path

notebook_path = Path(r"c:\Users\MadScie254\Documents\GitHub\Capstone-Einstein\Capstone-Einstein\notebooks\01_EDA_and_preprocessing.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Searching for cell with 'consumption_values < zero_threshold'...")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        for line in source:
            if "zeros_per_customer =" in line:
                print(f"Found in cell {i}:")
                print(repr(line))
