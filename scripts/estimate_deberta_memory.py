#!/usr/bin/env python3
"""
Estimate memory usage for DeBERTa-v3-base with filtered datasets
"""

def estimate_memory(model_name, max_aspects, batch_size):
    """
    Estimate GPU memory usage
    
    Args:
        model_name: 'bert-base' or 'deberta-v3-base'
        max_aspects: Maximum aspects in dataset
        batch_size: Training batch size
    """
    
    # Model base memory (approximate)
    model_memory = {
        'bert-base': 440,  # MB
        'deberta-v3-base': 550,  # MB (larger than BERT)
    }
    
    base_mem = model_memory.get(model_name, 500)
    
    # Pipeline does 6 forward passes per batch
    forward_passes = 6
    
    # Effective batch = batch_size * max_aspects
    effective_batch = batch_size * max_aspects
    
    # Total forward passes
    total_passes = effective_batch * forward_passes
    
    # Estimated memory (MB)
    estimated_mb = base_mem * total_passes
    estimated_gb = estimated_mb / 1024
    
    print("=" * 70)
    print(f"MEMORY ESTIMATION: {model_name}")
    print("=" * 70)
    print(f"Model base memory: {base_mem} MB")
    print(f"Max aspects: {max_aspects}")
    print(f"Batch size: {batch_size}")
    print(f"Effective batch: {batch_size} × {max_aspects} = {effective_batch}")
    print(f"Forward passes per batch: {forward_passes}")
    print(f"Total model calls: {effective_batch} × {forward_passes} = {total_passes}")
    print(f"\nEstimated GPU memory: ~{estimated_gb:.1f} GB")
    
    if estimated_gb < 12:
        print("✓ SAFE for 16GB GPU")
        return True
    elif estimated_gb < 15:
        print("⚠️  TIGHT - may work on 16GB GPU")
        return True
    else:
        print("❌ WILL CAUSE OOM on 16GB GPU")
        return False
    
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FILTERED DATASET (max_aspects = 4)")
    print("=" * 70 + "\n")
    
    # Test BERT
    print("BERT-base-uncased:")
    estimate_memory('bert-base', max_aspects=4, batch_size=2)
    print()
    
    # Test DeBERTa
    print("DeBERTa-v3-base:")
    safe = estimate_memory('deberta-v3-base', max_aspects=4, batch_size=2)
    print()
    
    if safe:
        print("=" * 70)
        print("✓ DeBERTa-v3-base should work with filtered datasets!")
        print("  Recommended: batch_size=2, max_aspects=4")
        print("=" * 70)
    else:
        print("=" * 70)
        print("⚠️  DeBERTa may still cause OOM")
        print("  Try: batch_size=1 or further filtering")
        print("=" * 70)
