#!/usr/bin/env python3
"""Test Phase 1: Data Conversion Quality"""

import json
import sys
from pathlib import Path

def test_span_reconstruction(original_jsonl, converted_json):
    """Test if token spans can reconstruct original text"""
    print("\n=== Test 1: Span Reconstruction ===")
    
    # Load data
    original = {}
    with open(original_jsonl, 'r') as f:
        for line in f:
            item = json.loads(line)
            original[item['ID']] = item
    
    converted = json.load(open(converted_json))
    
    total = 0
    perfect_match = 0
    errors = []
    
    for sample in converted:
        orig_id = sample['orig_id']
        if orig_id not in original:
            continue
        
        orig_item = original[orig_id]
        items = orig_item.get('Quadruplet', orig_item.get('Triplet', []))
        
        for item in items:
            aspect = item.get('Aspect')
            opinion = item.get('Opinion')
            
            if aspect == 'NULL' or opinion == 'NULL':
                continue
            
            total += 1
            
            # Find corresponding entities
            found_aspect = False
            found_opinion = False
            
            for i, entity in enumerate(sample['entities']):
                tokens = sample['tokens']
                reconstructed = " ".join(tokens[entity['start']:entity['end']])
                
                if entity['type'] == 'target' and reconstructed.lower() == aspect.lower():
                    found_aspect = True
                elif entity['type'] == 'opinion' and reconstructed.lower() == opinion.lower():
                    found_opinion = True
            
            if found_aspect and found_opinion:
                perfect_match += 1
            else:
                errors.append({
                    'id': orig_id,
                    'aspect': aspect,
                    'opinion': opinion,
                    'found_aspect': found_aspect,
                    'found_opinion': found_opinion
                })
    
    accuracy = (perfect_match / total * 100) if total > 0 else 0
    print(f"Total triplets: {total}")
    print(f"Perfect matches: {perfect_match}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if errors[:3]:
        print(f"\nSample errors (showing first 3):")
        for err in errors[:3]:
            print(f"  ID: {err['id']}")
            print(f"    Aspect: {err['aspect']} (found: {err['found_aspect']})")
            print(f"    Opinion: {err['opinion']} (found: {err['found_opinion']})")
    
    return {'total': total, 'matches': perfect_match, 'accuracy': accuracy, 'errors': len(errors)}

def test_va_preservation(original_jsonl, converted_json):
    """Test if VA scores are preserved correctly"""
    print("\n=== Test 2: VA Score Preservation ===")
    
    original = {}
    with open(original_jsonl, 'r') as f:
        for line in f:
            item = json.loads(line)
            original[item['ID']] = item
    
    converted = json.load(open(converted_json))
    
    total = 0
    preserved = 0
    va_errors = []
    
    for sample in converted:
        orig_id = sample['orig_id']
        if orig_id not in original:
            continue
        
        orig_item = original[orig_id]
        items = orig_item.get('Quadruplet', orig_item.get('Triplet', []))
        
        # Get VA scores from original
        orig_vas = [item['VA'] for item in items if item.get('Aspect') != 'NULL']
        
        # Get VA scores from converted
        conv_vas = [s['type'] for s in sample['sentiments']]
        
        for va in orig_vas:
            total += 1
            if va in conv_vas:
                preserved += 1
            else:
                va_errors.append({'id': orig_id, 'expected': va, 'found': conv_vas})
    
    accuracy = (preserved / total * 100) if total > 0 else 0
    print(f"Total VA scores: {total}")
    print(f"Preserved: {preserved}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return {'total': total, 'preserved': preserved, 'accuracy': accuracy}

def test_linguistic_features(converted_json):
    """Test if linguistic features are present"""
    print("\n=== Test 3: Linguistic Features ===")
    
    converted = json.load(open(converted_json))
    
    total = len(converted)
    has_pos = sum(1 for s in converted if s.get('pos'))
    has_dep = sum(1 for s in converted if s.get('dependency'))
    
    print(f"Total samples: {total}")
    print(f"Has POS tags: {has_pos} ({has_pos/total*100:.1f}%)")
    print(f"Has dependencies: {has_dep} ({has_dep/total*100:.1f}%)")
    
    # Check sample
    if converted:
        sample = converted[0]
        print(f"\nSample check (ID: {sample['orig_id']}):")
        print(f"  Tokens: {len(sample['tokens'])}")
        print(f"  POS tags: {len(sample['pos'])}")
        print(f"  Dependencies: {len(sample['dependency'])}")
        print(f"  Match: {len(sample['tokens']) == len(sample['pos']) == len(sample['dependency'])}")
    
    return {'total': total, 'has_pos': has_pos, 'has_dep': has_dep}

def test_statistics(converted_json):
    """Generate dataset statistics"""
    print("\n=== Test 4: Dataset Statistics ===")
    
    converted = json.load(open(converted_json))
    
    total_samples = len(converted)
    total_entities = sum(len(s['entities']) for s in converted)
    total_sentiments = sum(len(s['sentiments']) for s in converted)
    
    avg_entities = total_entities / total_samples if total_samples > 0 else 0
    avg_sentiments = total_sentiments / total_samples if total_samples > 0 else 0
    
    # Token length stats
    token_lengths = [len(s['tokens']) for s in converted]
    avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    min_tokens = min(token_lengths) if token_lengths else 0
    max_tokens = max(token_lengths) if token_lengths else 0
    
    print(f"Total samples: {total_samples}")
    print(f"Total entities: {total_entities}")
    print(f"Total sentiments: {total_sentiments}")
    print(f"Avg entities/sample: {avg_entities:.2f}")
    print(f"Avg sentiments/sample: {avg_sentiments:.2f}")
    print(f"Avg tokens/sample: {avg_tokens:.1f}")
    print(f"Token range: [{min_tokens}, {max_tokens}]")
    
    return {
        'samples': total_samples,
        'entities': total_entities,
        'sentiments': total_sentiments,
        'avg_entities': avg_entities,
        'avg_sentiments': avg_sentiments,
        'avg_tokens': avg_tokens
    }

def main():
    base_path = Path(__file__).parent.parent.parent
    
    # Test training data
    print("=" * 60)
    print("TESTING TRAINING DATA CONVERSION")
    print("=" * 60)
    
    original_train = base_path / "DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl"
    converted_train = base_path / "DESS/Codebase/data/dimabsa_eng_restaurant/train_dep_triple_polarity_result.json"
    
    results = {}
    results['span_test'] = test_span_reconstruction(original_train, converted_train)
    results['va_test'] = test_va_preservation(original_train, converted_train)
    results['features_test'] = test_linguistic_features(converted_train)
    results['stats'] = test_statistics(converted_train)
    
    # Test test data
    print("\n" + "=" * 60)
    print("TESTING TEST DATA CONVERSION")
    print("=" * 60)
    
    converted_test = base_path / "DESS/Codebase/data/dimabsa_eng_restaurant/test_dep_triple_polarity_result.json"
    results['test_stats'] = test_statistics(converted_test)
    results['test_features'] = test_linguistic_features(converted_test)
    
    # Save results
    output_file = base_path / "Testing/Phase1/test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)
    
    span_acc = results['span_test']['accuracy']
    va_acc = results['va_test']['accuracy']
    
    if span_acc >= 95 and va_acc >= 95:
        print("✅ PASS - Conversion quality is excellent")
        return 0
    elif span_acc >= 90 and va_acc >= 90:
        print("⚠️  PASS WITH WARNINGS - Conversion quality is acceptable")
        return 0
    else:
        print("❌ FAIL - Conversion quality needs improvement")
        return 1

if __name__ == '__main__':
    sys.exit(main())
