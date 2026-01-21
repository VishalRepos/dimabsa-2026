#!/usr/bin/env python3
"""
Filter predictions to keep only high-quality triplets
"""

import json
import sys

def filter_predictions(input_file, output_file, threshold=5.0):
    """
    Filter predictions based on VA magnitude threshold
    
    Args:
        input_file: Path to raw predictions
        output_file: Path to save filtered predictions
        threshold: Minimum VA magnitude to keep (default: 5.0)
    """
    
    # The input is actually the text output, need to extract JSON
    # Let's look for the actual JSON file
    
    print(f"Filtering predictions with threshold: {threshold}")
    
    # Try to find the actual JSON file
    json_file = input_file.replace('.txt', '.json')
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"JSON file not found: {json_file}")
        print("Please provide the path to the actual JSON file (laptop_predictions.json)")
        return
    
    filtered_data = []
    total_triplets_before = 0
    total_triplets_after = 0
    
    for sample in data:
        total_triplets_before += len(sample['Triplet'])
        
        # Filter triplets
        filtered_triplets = []
        for triplet in sample['Triplet']:
            # Parse VA scores
            va_str = triplet['VA']
            valence, arousal = map(float, va_str.split('#'))
            
            # Calculate magnitude
            magnitude = (valence**2 + arousal**2) ** 0.5
            
            # Keep if magnitude is above threshold
            # Also filter out single-character opinions and aspects
            if (magnitude >= threshold and 
                len(triplet['Aspect']) > 1 and 
                len(triplet['Opinion']) > 1 and
                triplet['Opinion'] not in [',', '.', '!', '?', '`', '-', '(', ')']):
                filtered_triplets.append(triplet)
        
        total_triplets_after += len(filtered_triplets)
        
        filtered_data.append({
            'ID': sample['ID'],
            'Triplet': filtered_triplets
        })
    
    # Save filtered data
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"\n✅ Filtering complete!")
    print(f"   Samples: {len(filtered_data)}")
    print(f"   Triplets before: {total_triplets_before:,}")
    print(f"   Triplets after: {total_triplets_after:,}")
    print(f"   Reduction: {(1 - total_triplets_after/total_triplets_before)*100:.1f}%")
    print(f"   Output: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python filter_predictions.py <input.json> <output.json> [threshold]")
        print("Example: python filter_predictions.py laptop_predictions.json laptop_filtered.json 5.0")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    
    filter_predictions(input_file, output_file, threshold)
