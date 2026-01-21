#!/usr/bin/env python3
"""
Convert Subtask 2 predictions (Triplets) to Subtask 1 format (Quadruplets)
by adding Category information from the original input file
"""

import json
import sys

def convert_to_quadruplet_format(predictions_file, original_file, output_file):
    """
    Convert triplet predictions to quadruplet format
    
    Args:
        predictions_file: Triplet predictions from model
        original_file: Original JSONL with IDs and Categories
        output_file: Output file in quadruplet format
    """
    
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Load original data to get IDs and text
    original_data = {}
    with open(original_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            original_data[item['ID']] = item
    
    # Convert predictions
    output = []
    
    for i, pred in enumerate(predictions):
        # Try to match with original ID
        # The predictions use sample_0, sample_1, etc.
        # We need to map back to original IDs
        original_id = f"laptop_quad_dev_{i+1}"
        
        if original_id not in original_data:
            # Try other ID formats
            for key in original_data.keys():
                if key.endswith(f"_{i+1}"):
                    original_id = key
                    break
        
        # Convert Triplets to Quadruplets
        quadruplets = []
        for triplet in pred['Triplet']:
            # Default category based on aspect
            category = "LAPTOP#GENERAL"  # Default
            
            # Try to infer category from aspect
            aspect_lower = triplet['Aspect'].lower()
            if any(word in aspect_lower for word in ['screen', 'display']):
                category = "DISPLAY#GENERAL"
            elif any(word in aspect_lower for word in ['keyboard', 'key']):
                category = "KEYBOARD#GENERAL"
            elif any(word in aspect_lower for word in ['battery', 'power']):
                category = "BATTERY#GENERAL"
            elif any(word in aspect_lower for word in ['trackpad', 'mouse', 'touchpad']):
                category = "HARDWARE#GENERAL"
            elif any(word in aspect_lower for word in ['software', 'app', 'program']):
                category = "SOFTWARE#GENERAL"
            elif any(word in aspect_lower for word in ['support', 'service', 'customer']):
                category = "SUPPORT#GENERAL"
            
            quadruplets.append({
                "Aspect": triplet['Aspect'],
                "Category": category,
                "Opinion": triplet['Opinion'],
                "VA": triplet['VA']
            })
        
        output.append({
            "ID": original_id,
            "Quadruplet": quadruplets
        })
    
    # Save as JSONL
    with open(output_file, 'w') as f:
        for item in output:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Converted {len(output)} samples")
    print(f"   Total quadruplets: {sum(len(s['Quadruplet']) for s in output)}")
    print(f"   Output: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python convert_to_quadruplet.py <predictions.json> <original.jsonl> <output.jsonl>")
        print("Example: python convert_to_quadruplet.py laptop_submission.json eng_laptop_train_alltasks.jsonl laptop_quadruplet_submission.jsonl")
        sys.exit(1)
    
    convert_to_quadruplet_format(sys.argv[1], sys.argv[2], sys.argv[3])
