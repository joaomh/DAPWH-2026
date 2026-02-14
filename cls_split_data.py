import os
import shutil
import splitfolders
from pathlib import Path

# --- IMAGES PATHs
BASE_DIR = Path('DAPWH')
VIEWS = ['Dorsal_Ventral', 'Frontal', 'Lateral']
MERGED_DIR = BASE_DIR / 'DAPWH_Merged_by_Family'
SPLIT_DIR = BASE_DIR / 'DAPWH_Final_Split'

def merge_and_split():
    # 1. Merge all views into a single family-based structure
    if not MERGED_DIR.exists():
        MERGED_DIR.mkdir(parents=True)
        
    for view in VIEWS:
        view_path = BASE_DIR / 'DAPWH' / view
        if not view_path.exists(): continue
        
        # Walk through each family folder in the current view
        for family_folder in view_path.iterdir():
            if family_folder.is_dir():
                target_family_dir = MERGED_DIR / family_folder.name
                target_family_dir.mkdir(exist_ok=True)
                
                # Copy images and prefix filename with view to avoid name collisions
                for img in family_folder.glob('*'):
                    if img.is_file():
                        new_name = f"{view}_{img.name}"
                        shutil.copy2(img, target_family_dir / new_name)

    print(f"Merge complete: {MERGED_DIR}")

    # 2. Split into Train (70%), Val (15%), Test (15%)
    splitfolders.ratio(
        str(MERGED_DIR), 
        output=str(SPLIT_DIR), 
        seed=42, 
        ratio=(.7, .15, .15), 
        move=False
    )
    print(f"Split complete: {SPLIT_DIR}")

if __name__ == "__main__":
    merge_and_split()

import os
import pandas as pd
from pathlib import Path

# path where the data were divded
SPLIT_DIR = Path('DAPWH_Final_Split')

def count_dataset_distribution(root_dir):
    data = []
    # train, val, test
    phases = ['train', 'val', 'test']
    
    # family classes
    families = sorted([d.name for d in (root_dir / 'train').iterdir() if d.is_dir()])
    
    for family in families:
        row = {'Family': family}
        total_family = 0
        
        for phase in phases:
            phase_family_path = root_dir / phase / family
            if phase_family_path.exists():
                count = len(list(phase_family_path.glob('*')))
                row[phase] = count
                total_family += count
            else:
                row[phase] = 0
        
        row['Total'] = total_family
        data.append(row)
    
    # dataframe to count each family
    df = pd.DataFrame(data)
    totals = df.select_dtypes(include=['number']).sum()
    df_total = pd.DataFrame([{'Family': 'total', **totals}])
    df = pd.concat([df, df_total], ignore_index=True)
    
    return df

distribution_df = count_dataset_distribution(SPLIT_DIR)

print("\nDataset Distribution by Family:")
print(distribution_df.to_string(index=False))