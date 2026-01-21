#!/usr/bin/env python3
"""
Analyze aspect distribution to determine optimal filtering threshold
"""

import json
import sys
from collections import Counter

def analyze_distribution(file_path):
    """Analyze aspect count distribution"""
    
    aspect_counts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'Quadruplet' in data:
                aspect_counts.append(len(data['Quadruplet']))
            elif 'Triplet' in data:
                aspect_counts.append(len(data['Triplet']))
    
    # Calculate statistics
    total = len(aspect_counts)
    counter = Counter(aspect_counts)
    
    print("=" * 70)
    print(f"ASPECT DISTRIBUTION ANALYSIS: {file_path.split('/')[-1]}")
    print("=" * 70)
    print(f"Total samples: {total}\n")
    
    print("Distribution:")
    print("-" * 70)
    cumulative = 0
    for count in sorted(counter.keys()):
        freq = counter[count]
        cumulative += freq
        pct = (freq / total) * 100
        cum_pct = (cumulative / total) * 100
        print(f"  {count:2d} aspects: {freq:4d} samples ({pct:5.2f}%) | Cumulative: {cum_pct:5.2f}%")
    
    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS:")
    print("=" * 70)
    
    for threshold in [3, 4, 5, 6, 7, 8]:
        kept = sum(counter[c] for c in counter if c <= threshold)
        removed = total - kept
        kept_pct = (kept / total) * 100
        removed_pct = (removed / total) * 100
        max_memory_gb = threshold * 6 * 0.44  # threshold * forward_passes * BERT_size
        
        print(f"\nThreshold = {threshold}:")
        print(f"  Samples kept: {kept:4d} ({kept_pct:5.2f}%)")
        print(f"  Samples removed: {removed:4d} ({removed_pct:5.2f}%)")
        print(f"  Estimated memory (batch=1): ~{max_memory_gb:.1f} GB")
        
        if max_memory_gb < 12:
            print(f"  ✓ SAFE for 16GB GPU")
        elif max_memory_gb < 15:
            print(f"  ⚠️  TIGHT but possible")
        else:
            print(f"  ❌ WILL CAUSE OOM")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)
    
    # Find optimal threshold (keeps >99% data, memory <12GB)
    for threshold in range(3, 20):
        kept = sum(counter[c] for c in counter if c <= threshold)
        kept_pct = (kept / total) * 100
        max_memory_gb = threshold * 6 * 0.44
        
        if kept_pct >= 99.0 and max_memory_gb < 12:
            print(f"Optimal threshold: {threshold}")
            print(f"  - Keeps {kept_pct:.2f}% of data")
            print(f"  - Memory: ~{max_memory_gb:.1f} GB")
            print(f"  - Safe for training ✓")
            return threshold
    
    print("Use threshold: 5 (conservative, safe choice)")
    return 5

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_distribution.py <path_to_data.jsonl>")
        sys.exit(1)
    
    analyze_distribution(sys.argv[1])
