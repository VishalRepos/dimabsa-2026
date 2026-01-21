#!/bin/bash
# Local Inference Setup and Execution

echo "============================================================"
echo "DimABSA Local Inference"
echo "============================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q torch transformers tqdm torch_geometric

# Run inference
echo ""
echo "============================================================"
echo "Running Inference"
echo "============================================================"
cd DESS/Codebase

python predict.py \
    --model_path ../../savedmodel \
    --test_data data/dimabsa_combined/test_dep_triple_polarity_result.json \
    --types_path data/types_va.json \
    --output ../../submission.json \
    --batch_size 8 \
    --device cpu

echo ""
echo "============================================================"
echo "Inference Complete!"
echo "============================================================"
echo "Submission file: submission.json"
echo ""
echo "To verify:"
echo "  python3 -c \"import json; d=json.load(open('submission.json')); print(f'Samples: {len(d)}'); print(f'Triplets: {sum(len(s[\\\"Triplet\\\"]) for s in d)}')\""
