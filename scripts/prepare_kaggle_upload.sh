#!/bin/bash
# Script to prepare Kaggle upload package

echo "=========================================="
echo "Preparing Kaggle Upload Package"
echo "=========================================="

# Set paths
BASE_DIR="/Users/vishal.thenuwara/Documents/MSC/Research/Coding/Competition/DimABSANew"
TEMP_DIR="/tmp/dimabsa-kaggle-upload"
OUTPUT_ZIP="/tmp/dimabsa-dess-data.zip"

# Clean up previous
rm -rf "$TEMP_DIR"
rm -f "$OUTPUT_ZIP"

# Create structure
echo "Creating directory structure..."
mkdir -p "$TEMP_DIR/DESS/Codebase/models"
mkdir -p "$TEMP_DIR/DESS/Codebase/trainer"
mkdir -p "$TEMP_DIR/DESS/Codebase/data/dimabsa_combined"

# Copy models
echo "Copying model files..."
cp "$BASE_DIR/DESS/Codebase/models/D2E2S_Model.py" "$TEMP_DIR/DESS/Codebase/models/"
cp "$BASE_DIR/DESS/Codebase/models/Syn_GCN.py" "$TEMP_DIR/DESS/Codebase/models/"
cp "$BASE_DIR/DESS/Codebase/models/Sem_GCN.py" "$TEMP_DIR/DESS/Codebase/models/"
cp "$BASE_DIR/DESS/Codebase/models/Attention_Module.py" "$TEMP_DIR/DESS/Codebase/models/"
cp "$BASE_DIR/DESS/Codebase/models/Channel_Fusion.py" "$TEMP_DIR/DESS/Codebase/models/"
cp "$BASE_DIR/DESS/Codebase/models/TIN_GCN.py" "$TEMP_DIR/DESS/Codebase/models/"
cp "$BASE_DIR/DESS/Codebase/models/General.py" "$TEMP_DIR/DESS/Codebase/models/"

# Copy trainer
echo "Copying trainer files..."
cp "$BASE_DIR/DESS/Codebase/trainer/loss.py" "$TEMP_DIR/DESS/Codebase/trainer/"
cp "$BASE_DIR/DESS/Codebase/trainer/input_reader.py" "$TEMP_DIR/DESS/Codebase/trainer/"
cp "$BASE_DIR/DESS/Codebase/trainer/entities.py" "$TEMP_DIR/DESS/Codebase/trainer/"
cp "$BASE_DIR/DESS/Codebase/trainer/util.py" "$TEMP_DIR/DESS/Codebase/trainer/"
cp "$BASE_DIR/DESS/Codebase/trainer/sampling.py" "$TEMP_DIR/DESS/Codebase/trainer/"
cp "$BASE_DIR/DESS/Codebase/trainer/evaluator.py" "$TEMP_DIR/DESS/Codebase/trainer/"

# Create __init__.py files
touch "$TEMP_DIR/DESS/Codebase/models/__init__.py"
touch "$TEMP_DIR/DESS/Codebase/trainer/__init__.py"

# Copy data
echo "Copying data files..."
cp "$BASE_DIR/DESS/Codebase/data/dimabsa_combined/train_dep_triple_polarity_result.json" "$TEMP_DIR/DESS/Codebase/data/dimabsa_combined/"
cp "$BASE_DIR/DESS/Codebase/data/dimabsa_combined/test_dep_triple_polarity_result.json" "$TEMP_DIR/DESS/Codebase/data/dimabsa_combined/"
cp "$BASE_DIR/DESS/Codebase/data/types_va.json" "$TEMP_DIR/DESS/Codebase/data/"

# Copy config
echo "Copying configuration files..."
cp "$BASE_DIR/DESS/Codebase/Parameter.py" "$TEMP_DIR/DESS/Codebase/"

# Create ZIP
echo "Creating ZIP file..."
cd "$TEMP_DIR"
zip -r "$OUTPUT_ZIP" DESS/ -q

# Check result
if [ -f "$OUTPUT_ZIP" ]; then
    SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)
    echo ""
    echo "=========================================="
    echo "✅ Package created successfully!"
    echo "=========================================="
    echo "Location: $OUTPUT_ZIP"
    echo "Size: $SIZE"
    echo ""
    echo "Next steps:"
    echo "1. Upload this ZIP to Kaggle as a dataset"
    echo "2. Upload kaggle_training.ipynb as a notebook"
    echo "3. Add the dataset to your notebook"
    echo "4. Run training!"
    echo ""
else
    echo "❌ Error: Failed to create ZIP file"
    exit 1
fi

# Cleanup
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "Done!"
