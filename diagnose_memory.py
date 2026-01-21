#!/usr/bin/env python3
"""
Diagnostic script to analyze memory usage in Pipeline-DeBERTa training
"""

import json
import sys

def analyze_dataset(file_path):
    """Analyze dataset to find max_aspect_num and memory implications"""
    
    max_aspects = 0
    total_samples = 0
    aspect_counts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            total_samples += 1
            
            # Count triplets/quadruplets (aspects)
            if 'Quadruplet' in data:
                num_aspects = len(data['Quadruplet'])
                aspect_counts.append(num_aspects)
                if num_aspects > max_aspects:
                    max_aspects = num_aspects
            elif 'Triplet' in data:
                num_aspects = len(data['Triplet'])
                aspect_counts.append(num_aspects)
                if num_aspects > max_aspects:
                    max_aspects = num_aspects
    
    avg_aspects = sum(aspect_counts) / len(aspect_counts) if aspect_counts else 0
    
    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    print(f"File: {file_path}")
    print(f"Total samples: {total_samples}")
    print(f"Max aspects in single sample: {max_aspects}")
    print(f"Average aspects per sample: {avg_aspects:.2f}")
    print()
    
    print("MEMORY IMPLICATIONS:")
    print("-" * 70)
    
    for batch_size in [1, 2, 4, 8]:
        effective_batch = batch_size * max_aspects
        forward_passes = 6  # f_asp, b_opi, f_opi, b_asp, valence, arousal
        total_model_calls = effective_batch * forward_passes
        
        print(f"\nBatch Size: {batch_size}")
        print(f"  Effective batch (with padding): {batch_size} × {max_aspects} = {effective_batch}")
        print(f"  Total model forward passes: {effective_batch} × {forward_passes} = {total_model_calls}")
        print(f"  Memory multiplier: {total_model_calls}x base model size")
        
        # Estimate memory for BERT-base
        bert_base_memory_mb = 440  # ~440MB for BERT-base model
        estimated_memory_gb = (bert_base_memory_mb * total_model_calls) / 1024
        print(f"  Estimated GPU memory (BERT-base): ~{estimated_memory_gb:.1f} GB")
        
        if estimated_memory_gb > 15:
            print(f"  ⚠️  WILL CAUSE OOM on T4/P100 (15-16GB)")
        elif estimated_memory_gb > 10:
            print(f"  ⚠️  HIGH RISK of OOM")
        else:
            print(f"  ✓ Should fit in GPU memory")
    
    print()
    print("=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print("1. Reduce max_aspect_num by filtering outliers")
    print("2. Use dynamic batching (don't pad all to max)")
    print("3. Process aspects sequentially instead of in parallel")
    print("4. Use gradient checkpointing")
    print("5. Consider mixed precision training (FP16)")
    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_memory.py <path_to_train_data.jsonl>")
        sys.exit(1)
    
    analyze_dataset(sys.argv[1])
