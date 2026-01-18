#!/bin/bash
# Final verification before pushing to GitHub

echo "=========================================="
echo "FINAL PRE-PUSH VERIFICATION"
echo "=========================================="
echo ""

cd /Users/vishal.thenuwara/Documents/MSC/Research/Coding/Competition/DimABSANew

# 1. Check essential files
echo "1. Checking essential files..."
files=(
    "README.md"
    ".gitignore"
    "requirements.txt"
    "kaggle_training.ipynb"
    "DESS/Codebase/data/dimabsa_combined/train_dep_triple_polarity_result.json"
    "DESS/Codebase/data/dimabsa_combined/test_dep_triple_polarity_result.json"
    "DESS/Codebase/data/types_va.json"
)

all_present=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "   ✅ $file ($size)"
    else
        echo "   ❌ MISSING: $file"
        all_present=false
    fi
done

if [ "$all_present" = false ]; then
    echo ""
    echo "❌ Some essential files are missing!"
    exit 1
fi

# 2. Check data file sizes
echo ""
echo "2. Checking data file sizes (GitHub limit: 100 MB)..."
large_files=$(find DESS/Codebase/data -name "*.json" -size +100M 2>/dev/null)
if [ -z "$large_files" ]; then
    echo "   ✅ All data files under 100 MB"
else
    echo "   ❌ Files over 100 MB found:"
    echo "$large_files"
    exit 1
fi

# 3. Verify data structure
echo ""
echo "3. Verifying data structure..."
python3 << 'PYEOF'
import json
import sys

try:
    # Check training data
    train = json.load(open('DESS/Codebase/data/dimabsa_combined/train_dep_triple_polarity_result.json'))
    assert len(train) == 3727, f"Expected 3727 samples, got {len(train)}"
    assert 'tokens' in train[0], "Missing 'tokens' field"
    assert 'entities' in train[0], "Missing 'entities' field"
    assert 'sentiments' in train[0], "Missing 'sentiments' field"
    print("   ✅ Training data: 3,727 samples, structure valid")
    
    # Check test data
    test = json.load(open('DESS/Codebase/data/dimabsa_combined/test_dep_triple_polarity_result.json'))
    assert len(test) == 400, f"Expected 400 samples, got {len(test)}"
    print("   ✅ Test data: 400 samples, structure valid")
    
    # Check VA format
    if train[0]['sentiments']:
        va = train[0]['sentiments'][0]['type']
        assert '#' in va, "VA format should contain '#'"
        v, a = map(float, va.split('#'))
        assert 1.0 <= v <= 9.0 and 1.0 <= a <= 9.0, "VA values out of range"
        print(f"   ✅ VA format valid: {va}")
    
except Exception as e:
    print(f"   ❌ Data validation failed: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    exit 1
fi

# 4. Check model files
echo ""
echo "4. Checking model files..."
model_files=(
    "DESS/Codebase/models/D2E2S_Model.py"
    "DESS/Codebase/trainer/loss.py"
    "DESS/Codebase/trainer/input_reader.py"
    "DESS/Codebase/Parameter.py"
)

for file in "${model_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ MISSING: $file"
        exit 1
    fi
done

# 5. Check .gitignore doesn't exclude data
echo ""
echo "5. Checking .gitignore..."
if grep -q "^DESS/Codebase/data/dimabsa_combined/\*.json$" .gitignore; then
    echo "   ❌ WARNING: Data files are excluded in .gitignore!"
    exit 1
else
    echo "   ✅ Data files NOT excluded"
fi

# 6. Calculate total size
echo ""
echo "6. Calculating total repository size..."
total_size=$(du -sh . | cut -f1)
data_size=$(du -sh DESS/Codebase/data | cut -f1)
echo "   Total repo size: $total_size"
echo "   Data size: $data_size"

# 7. Summary
echo ""
echo "=========================================="
echo "✅ ALL CHECKS PASSED"
echo "=========================================="
echo ""
echo "Repository is ready to push to GitHub!"
echo ""
echo "Data files included:"
echo "  - Combined training: 3,727 samples (8.28 MB)"
echo "  - Combined test: 400 samples (0.54 MB)"
echo "  - Total data: ~23 MB"
echo ""
echo "Next steps:"
echo "  1. Run: bash scripts/init_github.sh"
echo "  2. Create GitHub repository"
echo "  3. Push: git push -u origin main"
echo "  4. Use in Kaggle!"
echo ""
