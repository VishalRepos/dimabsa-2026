# GPU Utilization Analysis - Single GPU Usage

**Date**: 2026-01-22  
**Issue**: Only 1 GPU utilized on Kaggle T4 x2 setup  
**Status**: Expected behavior (not a bug)

---

## Observation

On Kaggle with T4 x2 (2 GPUs available), only 1 GPU is being utilized during training.

---

## Root Cause Analysis

### 1. Code Implementation

**File**: `run_task2&3_trainer_multilingual.py`

**Line 909-910**: Model initialization
```python
model = DimABSA(args.hidden_size, args.bert_model_type, len(category_mapping))
if args.gpu:
    model = model.cuda()  # ← Moves to default GPU (cuda:0)
```

**Issue**: 
- `model.cuda()` moves model to **default GPU only** (cuda:0)
- No multi-GPU wrapper (DataParallel or DistributedDataParallel)
- No explicit device specification

### 2. Missing Multi-GPU Support

**What's NOT in the code**:
```python
# Multi-GPU support NOT implemented:
# ❌ torch.nn.DataParallel(model)
# ❌ torch.nn.parallel.DistributedDataParallel(model)
# ❌ torch.cuda.device_count() check
# ❌ Device specification (cuda:0, cuda:1)
```

### 3. Batch Processing

**Current**: Sequential processing on single GPU
```python
for batch in batches:
    # All processing on cuda:0
    output = model(batch)  # GPU 0 only
```

**Not doing**: Parallel batch processing across GPUs

---

## Why Only 1 GPU is Used

### Technical Explanation

1. **Default Behavior**: `model.cuda()` uses `cuda:0` by default
2. **No Parallelization**: Code doesn't split work across GPUs
3. **Sequential Pipeline**: 6 forward passes done sequentially on same GPU
4. **No DataParallel**: PyTorch doesn't automatically use multiple GPUs

### Code Flow

```
Data → GPU 0 → Forward Pass 1 → GPU 0
              → Forward Pass 2 → GPU 0
              → Forward Pass 3 → GPU 0
              → Forward Pass 4 → GPU 0
              → Forward Pass 5 → GPU 0
              → Forward Pass 6 → GPU 0
              → Backward Pass → GPU 0

GPU 1: Idle (not utilized)
```

---

## Is This a Problem?

### Short Answer: **No, it's expected behavior**

### Reasons:

1. **Implementation Design**:
   - Code was written for single GPU
   - No multi-GPU logic implemented
   - Common for research code

2. **Memory Constraints**:
   - With filtered data: ~13GB on 1 GPU
   - Fits comfortably on single T4 (15GB)
   - Multi-GPU not needed for memory

3. **Batch Size = 1**:
   - Very small batch size
   - Not enough parallelism to benefit from 2 GPUs
   - Overhead would outweigh benefits

4. **Pipeline Architecture**:
   - 6 sequential forward passes
   - Hard to parallelize across GPUs
   - Would require significant refactoring

---

## Performance Impact

### Current (1 GPU):
- ✓ Works correctly
- ✓ Memory sufficient
- ✓ Training completes
- ⚠️ Slower than potential 2-GPU

### If Using 2 GPUs:
- Potential speedup: 1.5-1.8x (not 2x due to overhead)
- Requires code changes
- More complex to implement
- May not work with batch_size=1

---

## Solutions (If Multi-GPU Needed)

### Option 1: DataParallel (Easiest)

**Add after model creation**:
```python
model = DimABSA(args.hidden_size, args.bert_model_type, len(category_mapping))
if args.gpu:
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model = model.cuda()
```

**Pros**:
- Simple (2 lines of code)
- Automatic data splitting

**Cons**:
- Not efficient with batch_size=1
- Overhead with small batches
- May not help much

### Option 2: Manual GPU Assignment

**Split forward passes across GPUs**:
```python
# Forward passes 1-3 on GPU 0
with torch.cuda.device(0):
    output1 = model(batch1)
    output2 = model(batch2)
    output3 = model(batch3)

# Forward passes 4-6 on GPU 1
with torch.cuda.device(1):
    output4 = model(batch4)
    output5 = model(batch5)
    output6 = model(batch6)
```

**Pros**:
- Better control
- Can optimize for pipeline

**Cons**:
- Complex implementation
- Requires model refactoring
- Synchronization overhead

### Option 3: Increase Batch Size

**Use larger batches with DataParallel**:
```python
# Instead of batch_size=1
--batch_size 2  # Split across 2 GPUs (1 per GPU)
```

**Pros**:
- Better GPU utilization
- Faster training

**Cons**:
- May cause OOM (we're already at limit)
- Requires memory testing

---

## Recommendation

### For Current Setup: **Do Nothing**

**Reasons**:
1. ✓ Training works with 1 GPU
2. ✓ Memory is sufficient
3. ✓ batch_size=1 won't benefit from 2 GPUs
4. ✓ Implementation complexity not worth it
5. ✓ Focus on model performance, not training speed

### If Training is Too Slow:

**Better alternatives than multi-GPU**:
1. Use P100 instead of T4 (faster GPU)
2. Reduce epochs (3 → 2)
3. Use smaller model (BERT instead of DeBERTa)
4. Use gradient accumulation

---

## Verification

### Check GPU Usage on Kaggle

```python
import torch

print(f"GPUs available: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# During training
!nvidia-smi
```

**Expected output**:
```
GPUs available: 2
Current GPU: 0
GPU name: Tesla T4

nvidia-smi:
GPU 0: 13GB used (training)
GPU 1: 0GB used (idle)
```

---

## Comparison: Single vs Multi-GPU

| Aspect | Single GPU (Current) | Multi-GPU (Potential) |
|--------|---------------------|----------------------|
| **Implementation** | ✓ Simple | ❌ Complex |
| **Memory** | ✓ 13GB (sufficient) | Same per GPU |
| **Speed** | Baseline | 1.5-1.8x faster |
| **Batch Size** | 1 | Need 2+ for benefit |
| **Code Changes** | None | Significant |
| **Overhead** | None | Communication overhead |
| **Debugging** | ✓ Easy | ❌ Harder |
| **Recommendation** | ✓ Use this | Only if needed |

---

## Conclusion

**The single GPU usage is:**
- ✅ Expected behavior (not a bug)
- ✅ Correct for the implementation
- ✅ Sufficient for current needs
- ✅ Appropriate for batch_size=1

**Multi-GPU would require:**
- Significant code refactoring
- Larger batch sizes (currently impossible due to memory)
- Complex synchronization
- May not provide meaningful speedup

**Recommendation**: 
- Keep current single-GPU implementation
- Focus on model performance and accuracy
- If speed is critical, use P100 or reduce epochs

---

## Files to Modify (If Implementing Multi-GPU)

1. **run_task2&3_trainer_multilingual.py**
   - Line 909: Add DataParallel wrapper
   - Add GPU count check
   - Handle device placement

2. **DimABSAModel.py**
   - May need device-aware forward passes
   - Handle multi-GPU tensor operations

3. **Utils.py**
   - Update batch generation for multi-GPU
   - Handle device placement in loss calculation

**Estimated effort**: 4-6 hours of development + testing

---

**Status**: Single GPU usage is expected and appropriate for current configuration.
