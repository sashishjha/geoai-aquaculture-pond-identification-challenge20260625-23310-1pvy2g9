import sys
import numpy as np
import pandas as pd

sys.path.append('src')
import common as C

def main():
    te = pd.read_csv('data_checkpoint/raw/Test.csv')
    cube = C.band_cube(te)
    valid = C._valid_mask(cube)
    # S1 bands are indices 0, 1
    s1_valid = valid[:, :, 0]
    
    lengths = []
    starts = []
    for i in range(len(s1_valid)):
        row = s1_valid[i]
        months = np.where(row)[0]
        if len(months) > 0:
            starts.append(months[0])
            lengths.append(len(months))
            
    # Joint distribution of start and length
    from collections import Counter
    joint = Counter(zip(starts, lengths))
    print("Joint distribution of (start, length):")
    for k, v in sorted(joint.items()):
        print(f"  {k}: {v / len(starts):.4f}")
        
    print("\nLength distribution:")
    l_counts = Counter(lengths)
    for k, v in sorted(l_counts.items()):
        print(f"  {k}: {v / len(starts):.4f}")

if __name__ == '__main__':
    main()
