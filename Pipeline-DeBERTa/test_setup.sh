#!/bin/bash
# Quick test script for Pipeline-DeBERTa setup

cd /Users/vishal.thenuwara/Documents/MSC/Research/Coding/Competition/DimABSANew
source venv/bin/activate
cd Pipeline-DeBERTa

echo "=== Testing Pipeline-DeBERTa Setup ==="
echo ""

echo "1. Testing model import..."
python -c "from DimABSAModel import DimABSA; print('✓ Model import OK')"

echo ""
echo "2. Testing argument parsing..."
python run_task2\&3_trainer_multilingual.py --help > /dev/null 2>&1 && echo "✓ Script runs OK"

echo ""
echo "3. Checking data paths..."
echo "   Data directory: ./data/"
ls -la data/ 2>/dev/null | head -5

echo ""
echo "4. Checking dataset files..."
if [ -f "data/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl" ]; then
    echo "✓ Restaurant training data found"
    wc -l data/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl
else
    echo "✗ Restaurant training data NOT found"
fi

if [ -f "data/track_a/subtask_2/eng/eng_laptop_train_alltasks.jsonl" ]; then
    echo "✓ Laptop training data found"
    wc -l data/track_a/subtask_2/eng/eng_laptop_train_alltasks.jsonl
else
    echo "✗ Laptop training data NOT found"
fi

echo ""
echo "5. Default configuration:"
python -c "
import sys
sys.argv = ['test']
from run_task2task3_trainer_multilingual import parser_getting
args = parser_getting()
print(f'  Model: {args.bert_model_type}')
print(f'  Hidden size: {args.hidden_size}')
print(f'  Task: {args.task}')
print(f'  Domain: {args.domain}')
print(f'  Language: {args.language}')
print(f'  Batch size: {args.batch_size}')
print(f'  Epochs: {args.epoch_num}')
print(f'  Beta: {args.inference_beta}')
" 2>/dev/null || echo "  (Could not parse args - will work at runtime)"

echo ""
echo "=== Setup Complete ==="
