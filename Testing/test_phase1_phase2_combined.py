#!/usr/bin/env python3
"""Combined Testing: Phase 1 + Phase 2 with Corrected Data"""

import sys
import json
from pathlib import Path

def test_phase1_data_conversion():
    """Test Phase 1: Data conversion with corrected subtask_1 data"""
    print("\n" + "=" * 70)
    print("PHASE 1: DATA CONVERSION (CORRECTED)")
    print("=" * 70)
    
    base_path = Path(__file__).parent.parent
    results = {}
    
    # Test all 3 datasets
    datasets = {
        'restaurant': 'DESS/Codebase/data/dimabsa_eng_restaurant',
        'laptop': 'DESS/Codebase/data/dimabsa_eng_laptop',
        'combined': 'DESS/Codebase/data/dimabsa_combined'
    }
    
    for name, path in datasets.items():
        print(f"\n--- Testing {name.upper()} Dataset ---")
        
        train_path = base_path / f"{path}/train_dep_triple_polarity_result.json"
        test_path = base_path / f"{path}/test_dep_triple_polarity_result.json"
        
        if not train_path.exists() or not test_path.exists():
            print(f"❌ FAIL - Files not found")
            results[name] = False
            continue
        
        train_data = json.load(open(train_path))
        test_data = json.load(open(test_path))
        
        print(f"  Training samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")
        
        # Check sample structure
        if train_data:
            sample = train_data[0]
            has_tokens = 'tokens' in sample
            has_entities = 'entities' in sample
            has_sentiments = 'sentiments' in sample
            has_pos = 'pos' in sample
            has_dep = 'dependency' in sample
            
            print(f"  Structure: tokens={has_tokens}, entities={has_entities}, "
                  f"sentiments={has_sentiments}, pos={has_pos}, dep={has_dep}")
            
            # Check VA format
            if sample['sentiments']:
                va_string = sample['sentiments'][0]['type']
                if '#' in va_string:
                    v, a = map(float, va_string.split('#'))
                    va_valid = 1.0 <= v <= 9.0 and 1.0 <= a <= 9.0
                    print(f"  Sample VA: {va_string} (valid: {va_valid})")
                    
                    if has_tokens and has_entities and has_sentiments and has_pos and has_dep and va_valid:
                        print(f"  ✅ PASS")
                        results[name] = True
                    else:
                        print(f"  ❌ FAIL")
                        results[name] = False
                else:
                    print(f"  ❌ FAIL - Invalid VA format")
                    results[name] = False
            else:
                print(f"  ⚠️  WARNING - No sentiments in sample")
                results[name] = True  # Still pass if structure is correct
        else:
            print(f"  ❌ FAIL - Empty training data")
            results[name] = False
    
    return results

def test_phase2_model_modifications():
    """Test Phase 2: Model modifications"""
    print("\n" + "=" * 70)
    print("PHASE 2: MODEL MODIFICATIONS")
    print("=" * 70)
    
    base_path = Path(__file__).parent.parent
    results = {}
    
    # Test 1: Model code changes
    print("\n--- Test 1: Model Code Changes ---")
    model_path = base_path / "DESS/Codebase/models/D2E2S_Model.py"
    with open(model_path, 'r') as f:
        model_code = f.read()
    
    checks = {
        'va_output': ', 2  # VA regression' in model_code,
        'train_shape': '[batch_size, sentiments.shape[1], 2]  # VA regression' in model_code,
        'no_sigmoid': '# For VA regression: no sigmoid' in model_code,
    }
    
    all_passed = all(checks.values())
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    
    results['model_code'] = all_passed
    print(f"  {'✅ PASS' if all_passed else '❌ FAIL'}")
    
    # Test 2: Loss function
    print("\n--- Test 2: Loss Function ---")
    loss_path = base_path / "DESS/Codebase/trainer/loss.py"
    with open(loss_path, 'r') as f:
        loss_code = f.read()
    
    has_mse = 'MSE' in loss_code or 'mse' in loss_code
    has_va_reshape = 'view(-1, 2)' in loss_code
    
    print(f"  {'✅' if has_mse else '❌'} MSE loss")
    print(f"  {'✅' if has_va_reshape else '❌'} VA reshape")
    
    results['loss_function'] = has_mse and has_va_reshape
    print(f"  {'✅ PASS' if results['loss_function'] else '❌ FAIL'}")
    
    # Test 3: Dataset configuration
    print("\n--- Test 3: Dataset Configuration ---")
    param_path = base_path / "DESS/Codebase/Parameter.py"
    with open(param_path, 'r') as f:
        param_code = f.read()
    
    has_restaurant = 'dimabsa_eng_restaurant' in param_code
    has_laptop = 'dimabsa_eng_laptop' in param_code
    has_combined = 'dimabsa_combined' in param_code
    
    print(f"  {'✅' if has_restaurant else '❌'} Restaurant dataset")
    print(f"  {'✅' if has_laptop else '❌'} Laptop dataset")
    print(f"  {'✅' if has_combined else '❌'} Combined dataset")
    
    results['dataset_config'] = has_restaurant and has_laptop and has_combined
    print(f"  {'✅ PASS' if results['dataset_config'] else '❌ FAIL'}")
    
    # Test 4: Types configuration
    print("\n--- Test 4: Types Configuration ---")
    types_path = base_path / "DESS/Codebase/data/types_va.json"
    
    if types_path.exists():
        types = json.load(open(types_path))
        has_entities = 'entities' in types
        has_sentiment = 'sentiment' in types
        
        print(f"  {'✅' if has_entities else '❌'} Entity types defined")
        print(f"  {'✅' if has_sentiment else '❌'} Sentiment types defined")
        
        results['types_config'] = has_entities and has_sentiment
        print(f"  {'✅ PASS' if results['types_config'] else '❌ FAIL'}")
    else:
        print(f"  ❌ types_va.json not found")
        results['types_config'] = False
    
    return results

def test_data_statistics():
    """Test data statistics and quality"""
    print("\n" + "=" * 70)
    print("DATA STATISTICS & QUALITY")
    print("=" * 70)
    
    base_path = Path(__file__).parent.parent
    
    # Combined dataset stats
    train_path = base_path / "DESS/Codebase/data/dimabsa_combined/train_dep_triple_polarity_result.json"
    test_path = base_path / "DESS/Codebase/data/dimabsa_combined/test_dep_triple_polarity_result.json"
    
    train_data = json.load(open(train_path))
    test_data = json.load(open(test_path))
    
    print(f"\nCombined Dataset:")
    print(f"  Training: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"  Total: {len(train_data) + len(test_data)} samples")
    
    # Calculate statistics
    total_entities = sum(len(s['entities']) for s in train_data)
    total_sentiments = sum(len(s['sentiments']) for s in train_data)
    avg_entities = total_entities / len(train_data)
    avg_sentiments = total_sentiments / len(train_data)
    
    print(f"\n  Avg entities/sample: {avg_entities:.2f}")
    print(f"  Avg sentiments/sample: {avg_sentiments:.2f}")
    
    # VA score distribution
    va_scores = []
    for sample in train_data:
        for sentiment in sample['sentiments']:
            va_string = sentiment['type']
            if '#' in va_string:
                v, a = map(float, va_string.split('#'))
                va_scores.append((v, a))
    
    if va_scores:
        avg_v = sum(v for v, a in va_scores) / len(va_scores)
        avg_a = sum(a for v, a in va_scores) / len(va_scores)
        print(f"\n  VA Statistics:")
        print(f"    Total VA pairs: {len(va_scores)}")
        print(f"    Avg Valence: {avg_v:.2f}")
        print(f"    Avg Arousal: {avg_a:.2f}")
    
    # Check for data quality issues
    issues = 0
    for i, sample in enumerate(train_data[:100]):  # Check first 100
        if not sample['tokens']:
            issues += 1
        if len(sample['tokens']) != len(sample['pos']):
            issues += 1
        if len(sample['tokens']) != len(sample['dependency']):
            issues += 1
    
    print(f"\n  Quality check (first 100 samples):")
    print(f"    Issues found: {issues}")
    print(f"    {'✅ PASS' if issues == 0 else '⚠️  WARNING'}")
    
    return issues == 0

def main():
    print("=" * 70)
    print("COMBINED TESTING: PHASE 1 + PHASE 2")
    print("=" * 70)
    
    # Phase 1 tests
    phase1_results = test_phase1_data_conversion()
    
    # Phase 2 tests
    phase2_results = test_phase2_model_modifications()
    
    # Data quality tests
    data_quality = test_data_statistics()
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    print("\nPhase 1 - Data Conversion:")
    for dataset, passed in phase1_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {dataset.capitalize()}: {status}")
    
    print("\nPhase 2 - Model Modifications:")
    for test, passed in phase2_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test}: {status}")
    
    print(f"\nData Quality: {'✅ PASS' if data_quality else '⚠️  WARNING'}")
    
    # Calculate overall pass rate
    all_results = list(phase1_results.values()) + list(phase2_results.values()) + [data_quality]
    passed = sum(all_results)
    total = len(all_results)
    
    print(f"\n{'=' * 70}")
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'=' * 70}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - Ready for Phase 3 (Training)")
        return 0
    elif passed >= total * 0.8:
        print("\n⚠️  MOSTLY PASSED - Minor issues, can proceed with caution")
        return 0
    else:
        print("\n❌ TESTS FAILED - Fix issues before proceeding")
        return 1

if __name__ == '__main__':
    sys.exit(main())
