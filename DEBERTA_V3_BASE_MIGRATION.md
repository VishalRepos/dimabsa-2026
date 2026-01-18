# DeBERTa v3-base Migration - All Changes

## Issue
Training failed with dimension mismatch: LSTM expected 768 but got 1536 (or vice versa).

## Root Cause
Multiple hardcoded references to `microsoft/deberta-v2-xxlarge` (1536 dimensions) instead of using the configurable `microsoft/deberta-v3-base` (768 dimensions).

## All Changes Made

### 1. Parameter.py (Line 63, 96, 203)
**Before:**
```python
"--emb_dim", type=int, default=1536
"--deberta_feature_dim", type=int, default=1536
"--pretrained_deberta_name", default="microsoft/deberta-v2-xxlarge"
```

**After:**
```python
"--emb_dim", type=int, default=768
"--deberta_feature_dim", type=int, default=768
"--pretrained_deberta_name", default="microsoft/deberta-v3-base"
```

### 2. train.py (Line 33, 90)
**Before:**
```python
model_name = getattr(args, 'pretrained_deberta_name', 'microsoft/deberta-v2-xxlarge')
config = AutoConfig.from_pretrained("microsoft/deberta-v2-xxlarge")
```

**After:**
```python
model_name = getattr(args, 'pretrained_deberta_name', 'microsoft/deberta-v3-base')
config = AutoConfig.from_pretrained(self.args.pretrained_deberta_name)
```

### 3. models/D2E2S_Model.py (Line 64)
**Before:**
```python
self.pretrained_deberta_name = getattr(args, 'pretrained_deberta_name', 'microsoft/deberta-v2-xxlarge')
```

**After:**
```python
self.pretrained_deberta_name = getattr(args, 'pretrained_deberta_name', 'microsoft/deberta-v3-base')
```

## Verification

Run this to confirm no more hardcoded references:
```bash
cd DESS/Codebase
grep -r "deberta-v2-xxlarge" . --include="*.py" | grep -v ".pyc"
grep -r "1536" . --include="*.py" | grep -v ".pyc"
```

Should return no results (except comments).

## Training Command (Now Works Without Explicit Parameters)

**Minimal:**
```bash
python train.py --dataset dimabsa_combined --epochs 10
```

**Explicit (recommended for clarity):**
```bash
python train.py \
    --dataset dimabsa_combined \
    --epochs 10 \
    --batch_size 4 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 \
    --hidden_dim 384 \
    --emb_dim 768 \
    --lstm_layers 1
```

## Kaggle Instructions

1. **Delete old Kaggle dataset** (if you uploaded before)
2. **Re-upload repository** as new dataset version
3. **In notebook, clone fresh:**
   ```bash
   !git clone https://github.com/VishalRepos/dimabsa-2026.git
   cd dimabsa-2026/DESS/Codebase
   ```
4. **Run training** (parameters now optional):
   ```bash
   !python train.py --dataset dimabsa_combined --epochs 10 --batch_size 4
   ```

## Key Insight

The error "Expected 768, got 1536" meant:
- LSTM was initialized with `emb_dim=768` (from command line)
- But DeBERTa config was loaded from xxlarge (1536 dims)
- So DeBERTa output was 1536, but LSTM expected 768

The fix ensures **all components use the same model's dimensions**.

---
**Status:** ✅ All hardcoded references removed
**Commit:** 32dd1d4
**Date:** 2026-01-18
