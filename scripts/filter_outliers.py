#!/usr/bin/env python3
"""
Filter dataset to remove outlier samples with too many aspects
"""

import json
import sys
import os

def filter_dataset(input_path, output_path, max_aspects=4):
    """
    Filter dataset to keep only samples with <= max_aspects
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output filtered JSONL file
        max_aspects: Maximum number of aspects to keep (default: 4)
    """
    
    kept = 0
    removed = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            data = json.loads(line)
            
            # Count aspects
            num_aspects = 0
            if 'Quadruplet' in data:
                num_aspects = len(data['Quadruplet'])
            elif 'Triplet' in data:
                num_aspects = len(data['Triplet'])
            
            # Keep if within threshold
            if num_aspects <= max_aspects:
                fout.write(line)
                kept += 1
            else:
                removed += 1
    
    return kept, removed

def main():
    if len(sys.argv) < 3:
        print("Usage: python filter_outliers.py <input.jsonl> <output.jsonl> [max_aspects]")
        print("Example: python filter_outliers.py train.jsonl train_filtered.jsonl 4")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    max_aspects = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("FILTERING DATASET")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Max aspects threshold: {max_aspects}")
    print()
    
    kept, removed = filter_dataset(input_path, output_path, max_aspects)
    total = kept + removed
    kept_pct = (kept / total * 100) if total > 0 else 0
    removed_pct = (removed / total * 100) if total > 0 else 0
    
    print("Results:")
    print("-" * 70)
    print(f"Total samples: {total}")
    print(f"Kept: {kept} ({kept_pct:.2f}%)")
    print(f"Removed: {removed} ({removed_pct:.2f}%)")
    print()
    
    # Estimate memory
    memory_gb = max_aspects * 6 * 0.44
    print(f"Estimated memory (batch=1): ~{memory_gb:.1f} GB")
    
    if memory_gb < 12:
        print("✓ SAFE for 16GB GPU")
    elif memory_gb < 15:
        print("⚠️  TIGHT but possible")
    else:
        print("❌ May still cause OOM")
    
    print("=" * 70)
    print(f"✓ Filtered dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
