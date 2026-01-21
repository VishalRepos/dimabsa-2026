#!/usr/bin/env python
"""Test data loading for Pipeline-DeBERTa"""

import sys
import json
sys.path.insert(0, '.')

from DataProcess import dataset_process
from transformers import AutoTokenizer

def test_data_loading():
    print("=== Testing Data Loading ===\n")
    
    # Test configuration
    train_file = "data/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl"
    
    print(f"1. Loading data from: {train_file}")
    
    # Count lines
    with open(train_file, 'r') as f:
        lines = f.readlines()
        print(f"   Total samples: {len(lines)}")
    
    # Parse first sample
    sample = json.loads(lines[0])
    print(f"\n2. First sample structure:")
    print(f"   Keys: {list(sample.keys())}")
    print(f"   ID: {sample.get('ID', 'N/A')}")
    print(f"   Text: {sample.get('Text', 'N/A')[:80]}...")
    
    if 'Triplet' in sample:
        print(f"   Triplets: {len(sample['Triplet'])}")
        if sample['Triplet']:
            print(f"   First triplet: {sample['Triplet'][0]}")
    
    if 'Quadruplet' in sample:
        print(f"   Quadruplets: {len(sample['Quadruplet'])}")
    
    print(f"\n3. Testing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
    print(f"   Tokenizer loaded: {type(tokenizer).__name__}")
    
    # Test tokenization
    text = sample['Text']
    tokens = tokenizer.tokenize(text)
    print(f"   Text length: {len(text)} chars")
    print(f"   Token count: {len(tokens)}")
    print(f"   First 10 tokens: {tokens[:10]}")
    
    print(f"\n4. Testing dataset processing...")
    
    # Mock args for dataset_process
    class Args:
        task = 2
        max_len = 'max_len'
        max_aspect_num = 'max_aspect_num'
    
    args = Args()
    
    # Category mapping for restaurant
    restaurant_entity_labels = ['RESTAURANT', 'FOOD', 'DRINKS', 'AMBIENCE', 'SERVICE', 'LOCATION']
    restaurant_attribute_labels = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE_OPTIONS', 'MISCELLANEOUS']
    
    category_dict = {}
    category_list = []
    for entity in restaurant_entity_labels:
        for attribute in restaurant_attribute_labels:
            category = f"{entity}#{attribute}"
            category_dict[category] = len(category_list)
            category_list.append(category)
    
    print(f"   Category count: {len(category_list)}")
    
    try:
        # Process small subset
        all_data = []
        for i, line in enumerate(lines[:5]):  # Test first 5 samples
            data = json.loads(line)
            all_data.append(data)
        
        print(f"   Processing {len(all_data)} samples...")
        
        datasets_dict = {
            'train': all_data,
            'test': []
        }
        
        train_total_data = dataset_process(
            args=args,
            datatsets=datasets_dict,
            category_mapping=category_dict,
            tokenizer=tokenizer
        )
        
        print(f"   ✓ Processing successful!")
        print(f"   Train samples: {len(train_total_data['train'])}")
        print(f"   Max length: {train_total_data.get('max_len', 'N/A')}")
        print(f"   Max aspect num: {train_total_data.get('max_aspect_num', 'N/A')}")
        
        if train_total_data['train']:
            first_sample = train_total_data['train'][0]
            print(f"\n5. Processed sample structure:")
            print(f"   Type: {type(first_sample).__name__}")
            print(f"   Has forward_asp_query: {hasattr(first_sample, 'forward_asp_query')}")
            print(f"   Has valence_query: {hasattr(first_sample, 'valence_query')}")
            print(f"   Has arousal_query: {hasattr(first_sample, 'arousal_query')}")
        
        print(f"\n=== Data Loading Test PASSED ===")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
