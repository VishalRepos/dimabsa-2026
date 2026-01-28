# Pipeline-DeBERTa Inference & Submission

## Overview

Generate predictions using the trained Pipeline-DeBERTa model.

---

## Prerequisites

### 1. Trained Model
```bash
# Model should be saved from training
ls Pipeline-DeBERTa/model/
# Expected: task2_eng_res_best.pth (or similar)
```

### 2. Test Data
```bash
# Test data in DimABSA format (no conversion needed)
ls DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl
```

---

## Local Inference

### Basic Command
```bash
cd Pipeline-DeBERTa

python run_task2&3_trainer_multilingual.py \
  --task 2 \
  --domain res \
  --language eng \
  --train_data data/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
  --infer_data data/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode inference \
  --inference_beta 0.9
```

### Parameters
- `--mode inference`: Run inference only (no training)
- `--inference_beta`: Confidence threshold (0.0-1.0)
  - Higher = fewer but more confident predictions
  - Lower = more predictions but less confident
- `--train_data`: Still needed for loading model config
- `--infer_data`: Test data file

### Expected Output
```
Loading model from: model/task2_eng_res_best.pth
✓ Model loaded successfully

Loading test data...
✓ Loaded 200 samples

Running inference...
  Step 1: Extracting aspects (forward)...
  [====================] 25/25 batches
  
  Step 2: Extracting opinions (backward)...
  [====================] 25/25 batches
  
  Step 3: Extracting opinions (forward)...
  [====================] 25/25 batches
  
  Step 4: Extracting aspects (backward)...
  [====================] 25/25 batches
  
  Step 5: Predicting valence...
  [====================] 25/25 batches
  
  Step 6: Predicting arousal...
  [====================] 25/25 batches
  
  Combining results...
  ✓ Inference complete

Saving predictions...
✓ Predictions saved to: tasks/subtask_2/pred_eng_res.jsonl

Statistics:
  Total samples: 200
  Total triplets: 287
  Avg triplets/sample: 1.44
  Samples with no triplets: 18
```

---

## Kaggle Inference

### Step 1: Upload Trained Model
Same process as DESS (see [04-DESS-INFERENCE.md](04-DESS-INFERENCE.md))

### Step 2: Kaggle Notebook

**Cell 1 - Setup**:
```python
%cd /kaggle/working
!git clone https://github.com/YOUR-USERNAME/dimabsa-2026.git
%cd dimabsa-2026/Pipeline-DeBERTa

# Verify test data
!ls -lh ../DimABSA2026/task-dataset/track_a/subtask_2/eng/
```

**Cell 2 - Verify Model**:
```python
import os

model_path = "/kaggle/input/trainedpipeline"  # Your dataset name

if os.path.exists(model_path):
    print("✓ Model found!")
    !ls -lh {model_path}
else:
    print("✗ Model not found!")
```

**Cell 3 - Run Inference**:
```python
!python run_task2&3_trainer_multilingual.py \
  --task 2 \
  --domain res \
  --language eng \
  --train_data ../DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
  --infer_data ../DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode inference \
  --model_path /kaggle/input/trainedpipeline/task2_eng_res_best.pth \
  --inference_beta 0.9
```

**Cell 4 - Verify Submission**:
```python
import json

with open('tasks/subtask_2/pred_eng_res.jsonl', 'r') as f:
    lines = f.readlines()
    submission = [json.loads(line) for line in lines]

print(f"Total samples: {len(submission)}")
print(f"Total triplets: {sum(len(s['Triplet']) for s in submission)}")
print(f"\nFirst 3 predictions:")
for i in range(min(3, len(submission))):
    print(f"\n{i+1}. {json.dumps(submission[i], indent=2)}")
```

**Cell 5 - Copy to Output**:
```python
!cp tasks/subtask_2/pred_eng_res.jsonl /kaggle/working/submission.jsonl
!ls -lh /kaggle/working/submission.jsonl
```

---

## Inference Process

### Internal Pipeline

**Step 1: Aspect Extraction (Forward)**
```python
for sample in test_data:
    query = f"What is the aspect? {sample['Text']}"
    start_logits, end_logits = model(query, step='A')
    aspects = extract_spans(start_logits, end_logits, threshold=beta)
```

**Step 2: Opinion Extraction (Backward)**
```python
for aspect in aspects:
    query = f"What is the opinion about {aspect}? {sample['Text']}"
    start_logits, end_logits = model(query, step='OA')
    opinions = extract_spans(start_logits, end_logits, threshold=beta)
```

**Step 3: Opinion Extraction (Forward)**
```python
query = f"What is the opinion? {sample['Text']}"
start_logits, end_logits = model(query, step='O')
opinions_forward = extract_spans(start_logits, end_logits, threshold=beta)
```

**Step 4: Aspect Extraction (Backward)**
```python
for opinion in opinions_forward:
    query = f"What is the aspect about {opinion}? {sample['Text']}"
    start_logits, end_logits = model(query, step='AO')
    aspects_backward = extract_spans(start_logits, end_logits, threshold=beta)
```

**Step 5: Match and Pair**
```python
# Combine forward/backward extractions
all_aspects = merge(aspects, aspects_backward)
all_opinions = merge(opinions, opinions_forward)

# Create pairs
pairs = []
for aspect in all_aspects:
    for opinion in all_opinions:
        if is_valid_pair(aspect, opinion):
            pairs.append((aspect, opinion))
```

**Step 6: VA Prediction**
```python
for aspect, opinion in pairs:
    # Valence
    query = f"What is the valence of {aspect} {opinion}? {sample['Text']}"
    valence = model(query, step='Valence')
    
    # Arousal
    query = f"What is the arousal of {aspect} {opinion}? {sample['Text']}"
    arousal = model(query, step='Arousal')
    
    # Clip to valid range
    valence = clip(valence, 1.0, 9.0)
    arousal = clip(arousal, 1.0, 9.0)
    
    triplets.append({
        "Aspect": aspect,
        "Opinion": opinion,
        "VA": f"{valence:.2f}#{arousal:.2f}"
    })
```

**Step 7: Create Submission**
```python
submission.append({
    "ID": sample['ID'],
    "Triplet": triplets
})
```

---

## Tuning Inference

### Confidence Threshold (inference_beta)

**High Threshold (0.95)**:
- Fewer predictions
- Higher precision
- Lower recall
- Use when: You want high-quality predictions

**Medium Threshold (0.9)** - Default:
- Balanced predictions
- Good precision/recall trade-off
- Use when: Standard inference

**Low Threshold (0.8)**:
- More predictions
- Lower precision
- Higher recall
- Use when: You want to maximize coverage

### Example Comparison
```bash
# High precision
--inference_beta 0.95
# Result: 150 triplets, 85% precision, 60% recall

# Balanced
--inference_beta 0.9
# Result: 287 triplets, 75% precision, 75% recall

# High recall
--inference_beta 0.8
# Result: 450 triplets, 60% precision, 85% recall
```

---

## Output Format

### Submission File
**File**: `tasks/subtask_2/pred_eng_res.jsonl`

```json
{"ID": "restaurant_test_001", "Triplet": [{"Aspect": "sake list", "Opinion": "extensive", "VA": "7.85#7.98"}]}
{"ID": "restaurant_test_002", "Triplet": []}
{"ID": "restaurant_test_003", "Triplet": [{"Aspect": "food", "Opinion": "great", "VA": "7.50#7.62"}, {"Aspect": "service", "Opinion": "slow", "VA": "3.20#4.15"}]}
```

---

## Troubleshooting

### Issue 1: Model Not Found
```
FileNotFoundError: model/task2_eng_res_best.pth not found
```
**Solution**: Specify model path explicitly
```bash
--model_path model/task2_eng_res_epoch3.pth
```

### Issue 2: Empty Predictions
```
Total triplets: 0
```
**Solution**:
- Lower confidence threshold: `--inference_beta 0.8`
- Check model loaded correctly
- Verify test data format

### Issue 3: VA Out of Range
```
Validation Error: VA value 10.5 out of range
```
**Solution**: Check clipping in code
```python
valence = torch.clamp(valence, 1.0, 9.0)
arousal = torch.clamp(arousal, 1.0, 9.0)
```

### Issue 4: Mismatched Pairs
```
Aspect "food" paired with opinion "slow" (incorrect)
```
**Solution**:
- Improve pairing logic
- Add distance constraints
- Use dependency parsing (if available)

### Issue 5: Slow Inference
```
Inference taking > 30 minutes
```
**Solution**:
- Increase batch size: `--batch_size 16`
- Use GPU: Ensure CUDA available
- Reduce test set for debugging

---

## Submission to Competition

Same process as DESS (see [04-DESS-INFERENCE.md](04-DESS-INFERENCE.md)):

1. Download `submission.jsonl`
2. Go to Codabench
3. Upload to Subtask 2
4. Check results

---

## Performance Metrics

### Inference Speed
| GPU | Batch Size | Time (200 samples) |
|-----|------------|--------------------|
| T4 | 8 | ~8-10 min |
| P100 | 8 | ~5-7 min |
| V100 | 16 | ~3-5 min |
| CPU | 8 | ~30-40 min |

### Expected Results
- **Aspect Extraction F1**: ~75-80%
- **Opinion Extraction F1**: ~70-75%
- **Triplet F1**: ~10-15% (estimated)
- **Coverage**: ~85-90% of samples have at least one triplet

---

## Comparison with DESS Inference

| Aspect | Pipeline-DeBERTa | DESS |
|--------|------------------|------|
| **Inference Time** | 8-10 min | 2-3 min |
| **Forward Passes** | 6 per sample | 1 per sample |
| **Memory** | Higher | Lower |
| **Complexity** | Lower | Higher |
| **Debugging** | Easier | Harder |

---

## Next Steps

After inference:
1. Validate submission format
2. Submit to competition
3. Compare with DESS results
4. Consider ensemble approach

---

**Previous**: [06-PIPELINE-TRAINING.md](06-PIPELINE-TRAINING.md)  
**See Also**: [08-APPROACH-COMPARISON.md](08-APPROACH-COMPARISON.md)
