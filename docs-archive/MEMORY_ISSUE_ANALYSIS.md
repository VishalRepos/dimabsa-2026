# Memory Issue Analysis - Pipeline-DeBERTa/BERT

**Date**: 2026-01-21  
**Status**: 🔴 CRITICAL - Root cause identified

---

## Problem Statement

Both DeBERTa and BERT models cause Out of Memory (OOM) errors on Kaggle GPUs (T4 15GB, P100 16GB), even with batch_size=1.

---

## Root Cause

### The Issue is NOT the Model Size

The problem is **inefficient data padding** in the training pipeline:

1. **Outlier Sample**: One sample in the dataset has **26 aspects**
2. **Padding Strategy**: ALL samples are padded to `max_aspect_num = 26`
3. **Memory Multiplication**: Each batch processes `batch_size × max_aspect_num` samples
4. **Multiple Forward Passes**: 6 forward passes per batch (aspect, opinion, valence, arousal, etc.)

### Memory Calculation

**Restaurant Dataset**:
- Total samples: 2,284
- Max aspects: **26** (outlier)
- Average aspects: **1.60** (most samples have 1-2)

**With batch_size=1**:
```
Effective batch = 1 × 26 = 26 samples
Forward passes = 26 × 6 = 156 model calls
Memory needed = 156 × 440MB (BERT-base) = 67GB
Result: OOM on 16GB GPU ❌
```

**With batch_size=4**:
```
Effective batch = 4 × 26 = 104 samples
Forward passes = 104 × 6 = 624 model calls
Memory needed = 624 × 440MB = 268GB
Result: Catastrophic OOM ❌
```

---

## Code Analysis

### Location: `run_task2&3_trainer_multilingual.py`

**Training Loop (Lines 970-1100)**:
```python
for batch_index, batch_dict in enumerate(batch_generator):
    # 6 forward passes per batch:
    f_aspect_start_scores, f_aspect_end_scores = model(...)  # Pass 1
    b_opi_start_scores, b_opi_end_scores = model(...)        # Pass 2
    f_opi_start_scores, f_opi_end_scores = model(...)        # Pass 3
    b_asp_start_scores, b_asp_end_scores = model(...)        # Pass 4
    valence_scores = model(...)                               # Pass 5
    arousal_scores = model(...)                               # Pass 6
```

### Location: `DataProcess.py`

**Padding Logic (Line 385)**:
```python
for i in range(max_aspect_num - len(tokenized_QA.forward_opi_query)):
    # Pads ALL samples to max_aspect_num
    tokenized_QA.forward_opi_query.insert(-1, tokenized_QA.forward_opi_query[0])
```

**Max Calculation (Lines 652-661)**:
```python
max_aspect_num = 0
for line in dataset:
    if len(aspect_list) > max_aspect_num:
        max_aspect_num = len(aspect_list)  # Takes global maximum
```

---

## Why This Happens

1. **Pipeline Architecture**: The model processes aspects/opinions separately
2. **Batch Processing**: All aspects must be processed together for efficiency
3. **Static Padding**: Uses global max instead of dynamic per-batch max
4. **Outlier Impact**: One sample with 26 aspects forces ALL samples to pad to 26

---

## Solutions (Ranked by Feasibility)

### ✅ Solution 1: Filter Outliers (EASIEST)
**Remove samples with >5 aspects before training**

```python
# Add to data loading
filtered_data = [d for d in data if len(d['Quadruplet']) <= 5]
```

**Impact**:
- Max aspects: 26 → 5
- Memory: 67GB → 13GB (fits in 16GB GPU!)
- Data loss: Minimal (26 aspects is rare)

### ✅ Solution 2: Dynamic Batching (MODERATE)
**Calculate max_aspect_num per batch, not globally**

```python
# In generate_batches()
for batch in dataloader:
    batch_max = max(len(sample['aspects']) for sample in batch)
    # Pad only to batch_max, not global max
```

**Impact**:
- Memory: Varies per batch, much lower average
- Code changes: Moderate (modify DataProcess.py)

### ⚠️ Solution 3: Sequential Processing (COMPLEX)
**Process aspects one at a time instead of in parallel**

**Impact**:
- Memory: Minimal (no padding needed)
- Speed: Much slower (no parallelization)
- Code changes: Major refactoring

### ⚠️ Solution 4: Gradient Checkpointing (PARTIAL)
**Trade compute for memory**

**Impact**:
- Memory: ~30% reduction
- Speed: ~20% slower
- Still won't solve 67GB → 16GB gap

---

## Recommended Action Plan

### Immediate Fix (5 minutes)
1. Add outlier filtering to data loading
2. Set `max_aspect_threshold = 5`
3. Test with batch_size=2

### Short-term Fix (30 minutes)
1. Implement dynamic batching
2. Calculate per-batch max_aspect_num
3. Test with batch_size=4

### Long-term Fix (2-3 hours)
1. Refactor to sequential aspect processing
2. Remove global padding requirement
3. Optimize for variable-length inputs

---

## Testing Results

### Diagnostic Output
```
Dataset: eng_restaurant_train_alltasks.jsonl
Total samples: 2,284
Max aspects: 26
Average aspects: 1.60

Batch Size 1:
  Effective batch: 26
  Forward passes: 156
  Memory needed: 67GB
  Status: ⚠️ WILL CAUSE OOM
```

### Laptop Dataset
Run diagnostic:
```bash
python3 diagnose_memory.py DimABSA2026/task-dataset/track_a/subtask_1/eng/eng_laptop_train_alltasks.jsonl
```

---

## Files to Modify

1. **`run_task2&3_trainer_multilingual.py`**
   - Add outlier filtering in data loading (Line ~1190)

2. **`DataProcess.py`**
   - Modify `dataset_align()` for dynamic padding (Line 308)
   - Update `train_data_process()` to filter outliers (Line 703)

3. **`kaggle_pipeline_deberta.ipynb`**
   - Update documentation about memory requirements
   - Add outlier filtering step

---

## Next Steps

1. ✅ Run diagnostic on laptop dataset
2. ✅ Implement outlier filtering
3. ✅ Test with batch_size=2
4. ✅ Verify memory usage < 16GB
5. ✅ Train and validate results

---

**Conclusion**: The OOM issue is caused by inefficient padding, not model size. Filtering outliers or implementing dynamic batching will solve the problem.
