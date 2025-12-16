
import json
from pathlib import Path

notebook_path = Path(r"c:\Users\MadScie254\Documents\GitHub\Capstone-Einstein\Capstone-Einstein\notebooks\01_EDA_and_preprocessing.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Use partial match to be safe, but specific enough
target_snippet = "data_file = DATA_DIR / 'datasetsmall.csv'"

replacement_str = """# Load dataset (Big Data Mode)
data_file = DATA_DIR / 'dataset.csv'
if not data_file.exists():
    data_file = DATA_DIR / 'datasetsmall.csv'
    print('WARNING: Full dataset not found, falling back to small dataset')
else:
    print('Loading full dataset...')
df_raw = pd.read_csv(data_file)
"""

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if target_snippet in line:
                # We found the line. 
                # The replacement replaces the assignment AND the read_csv which is likely on the next line or same block
                # Let's inspect the next line to see if it's read_csv
                
                # Check if next line is read_csv
                if i+1 < len(source) and "pd.read_csv" in source[i+1]:
                    # Delete next line (read_csv) as it's included in our replacement block
                    del source[i+1]
                
                # Replace the target line with our block
                # We need to split our replacement string into a list of strings with newlines
                new_block = [s + '\n' for s in replacement_str.split('\n') if s]
                
                source[i:i+1] = new_block
                found = True
                print("Found and updated data source.")
                break
    if found:
        break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook saved successfully.")
else:
    print("Target line not found.")
