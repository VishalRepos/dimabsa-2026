#!/usr/bin/env python3
"""
Quick test of DESS model on DimABSA data
Tests: data loading, model initialization, memory usage
"""

import os
import sys
import torch

# Add DESS to path
sys.path.insert(0, '/kaggle/working/dimabsa-2026/DESS/Codebase' if os.path.exists('/kaggle') else './DESS/Codebase')

from Parameter import train_argparser
from models.D2E2S_Model import D2E2SModel
from trainer.input_reader import JsonInputReader
from trainer.entities import Dataset
import json

def quick_test():
    print("="*70)
    print("DESS QUICK TEST")
    print("="*70)
    
    # 1. Check GPU
    print("\n1. GPU Check:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 2. Load data
    print("\n2. Loading Data:")
    data_path = "./DESS/Codebase/data/dimabsa_combined/train_dep_triple_polarity_result.json"
    with open(data_path) as f:
        data = json.load(f)
    print(f"   Samples: {len(data)}")
    print(f"   Max entities: {max(len(d['entities']) for d in data)}")
    print(f"   Avg entities: {sum(len(d['entities']) for d in data) / len(data):.2f}")
    
    # 3. Check types
    print("\n3. Checking VA types:")
    types_path = "./DESS/Codebase/data/types_va.json"
    with open(types_path) as f:
        types = json.load(f)
    print(f"   Entity types: {types['entities']}")
    print(f"   Sentiment types: {len(types['sentiments'])} VA pairs")
    
    # 4. Memory estimation
    print("\n4. Memory Estimation:")
    max_entities = max(len(d['entities']) for d in data)
    model_mem_mb = 550  # DeBERTa-v3-base
    
    for batch_size in [1, 2, 4]:
        effective_batch = batch_size * max_entities
        forward_passes = 1  # DESS uses 1 pass
        mem_gb = (model_mem_mb * effective_batch * forward_passes) / 1024
        status = "✓ Safe" if mem_gb < 12 else "⚠️ Tight" if mem_gb < 15 else "❌ OOM"
        print(f"   Batch={batch_size}: {mem_gb:.1f} GB {status}")
    
    # 5. Test model initialization (if GPU available)
    if torch.cuda.is_available():
        print("\n5. Testing Model Initialization:")
        try:
            # Minimal args for testing
            class Args:
                pretrained_deberta_name = "microsoft/deberta-v3-base"
                max_span_size = 10
                prop_drop = 0.1
                freeze_transformer = False
                lstm_layers = 3
                lstm_drop = 0.4
                pos_size = 25
                char_lstm_layers = 1
                char_lstm_drop = 0.2
                char_size = 25
                use_glove = False
                use_pos = True
                use_char_lstm = True
                pool_type = "max"
                use_entity_ctx = True
                use_entity_span_lstm = True
                use_entity_common_lstm = True
                
            args = Args()
            
            # Load types for model
            with open(types_path) as f:
                types_data = json.load(f)
            
            print("   Creating model...")
            model = D2E2SModel(
                args=args,
                entity_types=len(types_data['entities']),
                sentiment_types=len(types_data['sentiments']),
                relation_types=0,
                prop_drop=args.prop_drop,
                freeze_transformer=args.freeze_transformer
            )
            
            if torch.cuda.is_available():
                model = model.cuda()
                print("   ✓ Model moved to GPU")
            
            # Check memory
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                print(f"   GPU Memory allocated: {mem_allocated:.2f} GB")
                print(f"   GPU Memory reserved: {mem_reserved:.2f} GB")
            
            print("   ✓ Model initialized successfully")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("="*70)
    
    max_entities = max(len(d['entities']) for d in data)
    mem_batch1 = (550 * max_entities * 1) / 1024
    
    if mem_batch1 < 12:
        print("✓ DESS should work with batch_size=1")
        print(f"  Estimated memory: {mem_batch1:.1f} GB")
        print("  Safe for 16GB GPU")
    else:
        print("⚠️ DESS may need filtering")
        print(f"  Current memory estimate: {mem_batch1:.1f} GB")
        print("  Consider filtering to max 15 entities")
    
    print("="*70)

if __name__ == "__main__":
    quick_test()
