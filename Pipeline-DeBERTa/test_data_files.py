#!/usr/bin/env python
"""Simple test to verify data files exist and are readable"""

import json
import sys

def test_data_files():
    print("=== Testing Data Files ===\n")
    
    files_to_test = [
        ("Restaurant Train", "data/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl"),
        ("Restaurant Dev", "data/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl"),
        ("Laptop Train", "data/track_a/subtask_2/eng/eng_laptop_train_alltasks.jsonl"),
        ("Laptop Dev", "data/track_a/subtask_2/eng/eng_laptop_dev_task2.jsonl"),
    ]
    
    all_ok = True
    for name, path in files_to_test:
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
                # Parse first line
                sample = json.loads(lines[0])
                
                print(f"✓ {name}:")
                print(f"  Path: {path}")
                print(f"  Samples: {len(lines)}")
                print(f"  Keys: {list(sample.keys())}")
                
                if 'Triplet' in sample:
                    print(f"  Triplets in first: {len(sample['Triplet'])}")
                if 'Quadruplet' in sample:
                    print(f"  Quadruplets in first: {len(sample['Quadruplet'])}")
                print()
                
        except Exception as e:
            print(f"✗ {name}: {e}\n")
            all_ok = False
    
    if all_ok:
        print("=== All Data Files OK ===")
        return True
    else:
        print("=== Some Files Failed ===")
        return False

if __name__ == "__main__":
    success = test_data_files()
    sys.exit(0 if success else 1)
