"""
Filter DimABSA dataset for Pipeline-DeBERTa training
Removes samples with too many triplets to avoid OOM
"""

import json
import argparse
from collections import Counter


def filter_dataset(input_file, output_file, max_triplets=4):
    """
    Filter dataset to keep only samples with <= max_triplets
    
    Args:
        input_file: Input JSONL file
        output_file: Output filtered JSONL file
        max_triplets: Maximum number of triplets per sample
    """
    
    filtered_data = []
    removed_count = 0
    triplet_counts = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # Count triplets/quadruplets
            num_triplets = len(data.get('Triplet', data.get('Quadruplet', [])))
            triplet_counts.append(num_triplets)
            
            # Keep if within limit
            if num_triplets <= max_triplets:
                filtered_data.append(data)
            else:
                removed_count += 1
    
    # Save filtered data
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in filtered_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    # Statistics
    print(f"\n{'='*70}")
    print(f"FILTERING RESULTS")
    print(f"{'='*70}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Max triplets allowed: {max_triplets}")
    print(f"\nOriginal samples: {len(triplet_counts)}")
    print(f"Filtered samples: {len(filtered_data)}")
    print(f"Removed samples: {removed_count}")
    print(f"Retention rate: {len(filtered_data)/len(triplet_counts)*100:.1f}%")
    
    print(f"\nTriplet distribution (original):")
    counter = Counter(triplet_counts)
    for count in sorted(counter.keys())[:10]:
        print(f"  {count} triplets: {counter[count]} samples")
    if len(counter) > 10:
        print(f"  ... ({len(counter)} unique counts total)")
    
    print(f"\nMax triplets in original: {max(triplet_counts)}")
    print(f"Max triplets in filtered: {max([len(d.get('Triplet', d.get('Quadruplet', []))) for d in filtered_data])}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', required=True, help='Output filtered JSONL file')
    parser.add_argument('--max_triplets', type=int, default=4, 
                       help='Maximum triplets per sample (default: 4)')
    args = parser.parse_args()
    
    filter_dataset(args.input, args.output, args.max_triplets)
    print("✓ Filtering complete!")
