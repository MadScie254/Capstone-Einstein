
import json
from pathlib import Path

notebook_path = Path(r"c:\Users\MadScie254\Documents\GitHub\Capstone-Einstein\Capstone-Einstein\notebooks\01_EDA_and_preprocessing.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "zeros_per_customer = np.sum(consumption_values < zero_threshold, axis=1)" in line:
                # Keep indentation if any
                indentation = line[:line.find("zeros_per_customer")]
                source[i] = indentation + "zeros_per_customer = np.sum(consumption_values < zero_threshold, axis=1).astype(float)\n"
                found = True
                print("Found and fixed the line.")
                break
    if found:
        break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook saved successfully.")
else:
    print("Target line not found.")
