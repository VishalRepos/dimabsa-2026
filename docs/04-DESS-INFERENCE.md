# DESS Inference & Submission

## Overview

Generate predictions on test data and create submission file for competition.

---

## Prerequisites

### 1. Trained Model
```bash
# Model should be saved from training
ls savemodels/dimabsa_combined/
# Expected: best_model.pt, config.json, tokenizer files
```

### 2. Test Data
```bash
# Test data in DESS format
ls DESS/Codebase/data/dimabsa_combined/test_dep_triple_polarity_result.json
```

---

## Local Inference

### Basic Command
```bash
cd DESS/Codebase

python predict.py \
    --model_path savemodels/dimabsa_combined \
    --test_data data/dimabsa_combined/test_dep_triple_polarity_result.json \
    --types_path data/types_va.json \
    --output submission.json \
    --batch_size 4 \
    --device cuda
```

### Parameters
- `--model_path`: Directory containing trained model
- `--test_data`: Test data in DESS format
- `--types_path`: Entity/sentiment type definitions
- `--output`: Output file path (JSONL format)
- `--batch_size`: Batch size (use same as training: 4)
- `--device`: `cuda` or `cpu`

### Expected Output
```
Loading model from savemodels/dimabsa_combined...
✓ Model loaded successfully

Loading test data...
✓ Loaded 400 samples

Running inference...
[====================] 100/100 batches
✓ Inference complete

Generating submission...
✓ Submission saved to submission.json

Statistics:
  Total samples: 400
  Total triplets: 587
  Avg triplets/sample: 1.47
  Samples with no triplets: 43
```

---

## Kaggle Inference

### Step 1: Upload Trained Model

1. **Compress Model**:
```bash
cd savemodels/dimabsa_combined
zip -r trained_model.zip .
```

2. **Upload to Kaggle**:
   - Kaggle → Datasets → New Dataset
   - Upload `trained_model.zip`
   - Name: `traineddess` (or your choice)
   - Make public or private

### Step 2: Create Inference Notebook

**Cell 1 - Setup**:
```python
%cd /kaggle/working
!rm -rf dimabsa-2026  # Clean previous runs
!git clone https://github.com/YOUR-USERNAME/dimabsa-2026.git
%cd dimabsa-2026/DESS/Codebase

# Verify test data
!ls -lh data/dimabsa_combined/test_dep_triple_polarity_result.json
```

**Cell 2 - Verify Model**:
```python
import os

model_path = "/kaggle/input/traineddess"  # Update with your dataset name

print(f"Looking for model at: {model_path}")
if os.path.exists(model_path):
    print("\n✓ Model found!")
    !ls -lh {model_path}
else:
    print("\n✗ Model not found!")
    print("Available datasets:")
    !ls /kaggle/input/
```

**Cell 3 - Run Inference**:
```python
!python predict.py \
    --model_path /kaggle/input/traineddess \
    --test_data data/dimabsa_combined/test_dep_triple_polarity_result.json \
    --types_path data/types_va.json \
    --output /kaggle/working/submission.json \
    --batch_size 4 \
    --device cuda
```

**Cell 4 - Verify Submission**:
```python
import json

with open('/kaggle/working/submission.json', 'r') as f:
    submission = json.load(f)

print(f"Total samples: {len(submission)}")
print(f"Total triplets: {sum(len(s['Triplet']) for s in submission)}")
print(f"Samples with triplets: {sum(1 for s in submission if s['Triplet'])}")
print(f"\nFirst 3 predictions:")
for i in range(min(3, len(submission))):
    print(f"\n{i+1}. ID: {submission[i]['ID']}")
    print(f"   Triplets: {len(submission[i]['Triplet'])}")
    if submission[i]['Triplet']:
        print(f"   Example: {submission[i]['Triplet'][0]}")
```

**Cell 5 - Download**:
```python
# File will be available in Output panel
!ls -lh /kaggle/working/submission.json
print("\n✓ Download from Output panel →")
```

---

## Prediction Process

### Internal Flow
```python
# 1. Load model
model = D2E2SModel(...)
model.load_state_dict(checkpoint)
model.eval()

# 2. For each test sample:
with torch.no_grad():
    # Encode
    h = model.deberta(tokens)
    h = model.lstm(h)
    
    # Dual-channel GCN
    h = model.fusion(h, h_syn, h_sem)
    
    # Predict entities
    entity_clf = model._classify_entities(h)
    entity_spans = model._filter_spans(entity_clf)
    
    # Generate pairs
    pairs = [(aspect, opinion) for aspect in aspects for opinion in opinions]
    
    # Predict VA for each pair
    va_scores = model._classify_sentiments(h, pairs)
    
    # Filter by confidence
    valid_pairs = [(a, o, v, ar) for (a, o), (v, ar) in zip(pairs, va_scores) 
                   if confidence > threshold]

# 3. Convert to DimABSA format
triplets = []
for aspect_span, opinion_span, valence, arousal in valid_pairs:
    aspect_text = " ".join(tokens[aspect_span[0]:aspect_span[1]])
    opinion_text = " ".join(tokens[opinion_span[0]:opinion_span[1]])
    va_string = f"{valence:.2f}#{arousal:.2f}"
    
    triplets.append({
        "Aspect": aspect_text,
        "Opinion": opinion_text,
        "VA": va_string
    })

# 4. Create submission
submission.append({
    "ID": sample_id,
    "Triplet": triplets
})
```

---

## Submission Validation

### Automatic Validation
```python
# Built into predict.py
def validate_submission(submission):
    for sample in submission:
        # Check required fields
        assert "ID" in sample
        assert "Triplet" in sample
        
        # Check triplet format
        for triplet in sample["Triplet"]:
            assert "Aspect" in triplet
            assert "Opinion" in triplet
            assert "VA" in triplet
            
            # Check VA format
            va = triplet["VA"]
            assert "#" in va
            valence, arousal = map(float, va.split("#"))
            assert 1.0 <= valence <= 9.0
            assert 1.0 <= arousal <= 9.0
```

### Manual Validation Script
```bash
python scripts/validate_submission.py submission.json
```

---

## Submission to Competition

### Step 1: Download Submission File
- From Kaggle Output panel
- Or from local inference

### Step 2: Upload to Codabench
1. Go to: https://www.codabench.org/competitions/10918/
2. Click "Submit / View Results"
3. Select "Subtask 2"
4. Upload `submission.json`
5. Add description (optional)
6. Click "Submit"

### Step 3: Check Results
- Wait for evaluation (~2-5 minutes)
- Check leaderboard for your score
- View detailed results (if available)

---

## Troubleshooting

### Issue 1: Model Path Error
```
FileNotFoundError: model.safetensors not found
```
**Solution**: Use folder path, not file path
```python
# Wrong:
--model_path /kaggle/input/traineddess/model.safetensors

# Correct:
--model_path /kaggle/input/traineddess
```

### Issue 2: Batch Size Mismatch
```
RuntimeError: Expected hidden size (2, 4, 384), got (2, 8, 384)
```
**Solution**: Use same batch size as training
```bash
--batch_size 4  # Same as training
```

### Issue 3: Empty Predictions
```
Total triplets: 0
```
**Solution**:
- Check confidence threshold (lower it)
- Verify model loaded correctly
- Check test data format

### Issue 4: Invalid VA Format
```
Validation Error: VA format invalid
```
**Solution**: Check VA string formatting in predict.py
```python
va_string = f"{valence:.2f}#{arousal:.2f}"  # Must be 2 decimals
```

### Issue 5: Missing Test IDs
```
Submission Error: Missing IDs
```
**Solution**: Ensure all test IDs are in submission
```python
# Check test data IDs match submission IDs
test_ids = set(sample['ID'] for sample in test_data)
submission_ids = set(sample['ID'] for sample in submission)
assert test_ids == submission_ids
```

---

## Output Format

### Submission File Structure
```json
[
  {
    "ID": "restaurant_test_001",
    "Triplet": [
      {
        "Aspect": "sake list",
        "Opinion": "extensive",
        "VA": "7.85#7.98"
      }
    ]
  },
  {
    "ID": "restaurant_test_002",
    "Triplet": []
  },
  {
    "ID": "restaurant_test_003",
    "Triplet": [
      {
        "Aspect": "food",
        "Opinion": "great",
        "VA": "7.50#7.62"
      },
      {
        "Aspect": "service",
        "Opinion": "slow",
        "VA": "3.20#4.15"
      }
    ]
  }
]
```

### Statistics to Check
- Total samples: Should match test set size (400)
- Total triplets: Typically 400-800
- Avg triplets/sample: ~1.0-2.0
- Samples with no triplets: ~10-20%
- VA range: All values in [1.00, 9.00]

---

## Performance Metrics

### Inference Speed
| GPU | Batch Size | Time (400 samples) |
|-----|------------|--------------------|
| T4 | 4 | ~2-3 min |
| P100 | 4 | ~1-2 min |
| V100 | 8 | ~1 min |
| CPU | 4 | ~15-20 min |

### Expected Results
- **Entity F1**: ~75-80% (aspect/opinion detection)
- **Continuous F1**: ~8-10% (full triplet with VA)
- **Coverage**: ~80-90% of samples have at least one triplet

---

## Next Steps

After submission:
1. Check leaderboard score
2. Analyze errors (if detailed results available)
3. Iterate on model/training if needed
4. Consider ensemble with Pipeline-DeBERTa

---

**Previous**: [03-DESS-TRAINING.md](03-DESS-TRAINING.md)  
**See Also**: [08-APPROACH-COMPARISON.md](08-APPROACH-COMPARISON.md)
