# Phase 3: Training - Kaggle Setup Complete

## Summary

Phase 3 training materials prepared for Kaggle execution.

---

## Files Created

### 1. Training Notebook
**File**: `kaggle_training.ipynb`

**Features**:
- Complete training pipeline
- GPU-optimized for Kaggle
- Progress monitoring with tqdm
- Automatic checkpoint saving
- Training curve visualization
- Model evaluation

**Sections**:
1. Environment setup
2. Data loading and verification
3. Training configuration
4. Model initialization
5. Optimizer and loss setup
6. Training loop with progress bars
7. Training history and visualization
8. Model evaluation
9. Download instructions

### 2. Setup Guide
**File**: `KAGGLE_SETUP_GUIDE.md`

**Contents**:
- Step-by-step Kaggle setup
- File preparation instructions
- Dataset upload guide
- Notebook configuration
- Troubleshooting tips
- Expected outputs

### 3. Upload Package Script
**File**: `scripts/prepare_kaggle_upload.sh`

**Function**: Automatically creates ZIP file with:
- Model files (7 files)
- Trainer files (6 files)
- Data files (combined dataset)
- Configuration files
- Proper directory structure

**Output**: `/tmp/dimabsa-dess-data.zip` (820 KB)

---

## Training Configuration

### Model
- **Architecture**: DESS with VA regression
- **Base Model**: DeBERTa-v3-base (faster for Kaggle)
- **Output**: 2 values (valence, arousal)
- **Loss**: MSE for VA regression

### Hyperparameters
```python
{
    'batch_size': 4,
    'epochs': 10,
    'learning_rate': 5e-5,
    'max_grad_norm': 1.0,
    'warmup_proportion': 0.1,
    'weight_decay': 0.01,
}
```

### Dataset
- **Training**: 3,727 samples (restaurant + laptop)
- **Test**: 400 samples
- **Batches per epoch**: ~932
- **Total training steps**: ~9,320

---

## Kaggle Setup Steps

### Step 1: Prepare Upload Package ✅
```bash
bash scripts/prepare_kaggle_upload.sh
```
**Output**: `/tmp/dimabsa-dess-data.zip` (820 KB)

### Step 2: Upload to Kaggle
1. Go to https://www.kaggle.com/datasets
2. Create new dataset
3. Upload `dimabsa-dess-data.zip`
4. Title: "DimABSA DESS Training Data"
5. Make private

### Step 3: Create Notebook
1. Go to https://www.kaggle.com/code
2. Create new notebook
3. Upload `kaggle_training.ipynb`
4. Add dataset to notebook
5. Enable GPU (T4 x2 or P100)

### Step 4: Run Training
1. Update `DATA_PATH` in notebook
2. Click "Run All"
3. Monitor progress (~2-3 hours)

### Step 5: Download Results
- Trained model: `/kaggle/working/checkpoints/best_model.pt`
- Training history: `/kaggle/working/logs/training_history.json`
- Training curve: `/kaggle/working/logs/training_curve.png`

---

## Expected Training Time

| GPU | Time per Epoch | Total Time (10 epochs) |
|-----|----------------|------------------------|
| T4 x2 | ~15-20 min | ~2.5-3 hours |
| P100 | ~10-12 min | ~1.5-2 hours |
| V100 | ~8-10 min | ~1-1.5 hours |

---

## Training Monitoring

### Progress Display
```
Epoch 1/10: 100%|██████████| 932/932 [15:23<00:00]
  Avg Loss: 2.3456
  Time: 923.45s
  ✅ Best model saved (loss: 2.3456)
```

### Metrics Tracked
- Training loss per epoch
- Time per epoch
- Best model checkpoint
- Training curve visualization

---

## Output Files

### 1. Trained Model
**File**: `best_model.pt`
**Size**: ~1.5 GB
**Contains**:
- Model state dict
- Optimizer state
- Epoch number
- Best loss value

### 2. Training History
**File**: `training_history.json`
**Format**:
```json
[
  {"epoch": 1, "loss": 2.3456, "time": 923.45},
  {"epoch": 2, "loss": 1.8234, "time": 920.12},
  ...
]
```

### 3. Training Curve
**File**: `training_curve.png`
**Content**: Loss vs Epoch plot

---

## Troubleshooting

### Out of Memory
**Solution**: Reduce batch_size from 4 to 2

### Slow Training
**Check**: GPU is enabled in settings

### Dataset Not Found
**Fix**: Verify dataset path in notebook

---

## Next Steps After Training

### Phase 4: Inference
1. Download trained model from Kaggle
2. Create inference script
3. Generate predictions on test data
4. Convert to submission format
5. Submit to DimABSA competition

### Expected Model Performance
- **Task**: Subtask 2 (Triplet Extraction)
- **Output**: (Aspect, Opinion, VA)
- **Metric**: Continuous F1
- **Target**: Competitive baseline

---

## Files Ready for Kaggle

```
✅ kaggle_training.ipynb          # Training notebook
✅ /tmp/dimabsa-dess-data.zip     # Data package (820 KB)
✅ KAGGLE_SETUP_GUIDE.md          # Setup instructions
✅ scripts/prepare_kaggle_upload.sh  # Package creator
```

---

## Validation Checklist

- [x] Training notebook created
- [x] Data package prepared (820 KB)
- [x] Setup guide written
- [x] Upload script tested
- [x] GPU optimization configured
- [x] Progress monitoring included
- [x] Checkpoint saving implemented
- [x] Visualization added
- [x] Download instructions provided

---

## Ready to Train!

All materials prepared for Kaggle training. Follow the setup guide to:
1. Upload data package
2. Create notebook
3. Run training
4. Download results

**Estimated total time**: 3-4 hours (including setup)

---

*Phase 3 prepared: 2026-01-18*
