# ✅ Kaggle Training Notebook Ready!

## New File Created

**File**: `kaggle_training_final.ipynb`

**Based on**: `Kaggle_example.ipynb` structure

**Status**: ✅ Pushed to GitHub

---

## What's Different from Example

### Example Notebook (Kaggle_example.ipynb)
- DESS-improved repository
- 14res dataset (original ASTE task)
- Sentiment classification (POS/NEG/NEU)
- Cross-attention fusion experiments

### Our Notebook (kaggle_training_final.ipynb)
- ✅ dimabsa-2026 repository (your repo)
- ✅ dimabsa_combined dataset (3,727 samples)
- ✅ VA regression (continuous scores)
- ✅ MSE loss for VA prediction
- ✅ Simplified for direct training

---

## How to Use in Kaggle

### Step 1: Create Notebook
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Title: "DimABSA VA Regression Training"

### Step 2: Upload Notebook
1. File → Upload Notebook
2. Select `kaggle_training_final.ipynb`
3. Notebook will load with all cells

### Step 3: Configure
1. Settings → Accelerator → **GPU T4 x2** (or P100)
2. Settings → Internet → **ON** (to clone GitHub)
3. Settings → Persistence → **Files only**

### Step 4: Run Training
1. Click **"Run All"**
2. Wait for completion (~2-3 hours)
3. Download results

---

## Notebook Structure

### Cell 1: Clone Repository
```bash
git clone https://github.com/VishalRepos/dimabsa-2026.git
cd dimabsa-2026/DESS/Codebase
```

### Cell 2: Check GPU
```bash
nvidia-smi
```

### Cell 3: Install Dependencies
```bash
pip install torch transformers numpy scikit-learn tqdm...
```

### Cell 4: Verify Setup
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
```

### Cell 5: Verify Data
```python
train_data = json.load(open('data/dimabsa_combined/train_dep_triple_polarity_result.json'))
print(f"Training samples: {len(train_data)}")  # 3727
```

### Cell 6: Training Config
```python
DATASET = "dimabsa_combined"
EPOCHS = 10
BATCH_SIZE = 4
```

### Cell 7: Run Training ⭐
```bash
python train.py \
    --dataset dimabsa_combined \
    --epochs 10 \
    --batch_size 4 \
    --pretrained_deberta_name microsoft/deberta-v3-base
```

### Cell 8-11: View Results & Download

---

## Expected Output

### During Training
```
Epoch 1/10: [████████] 932/932
  Loss: 2.3456
  Time: 15:23

Epoch 2/10: [████████] 932/932
  Loss: 1.8234
  Time: 15:20
...
```

### After Training
```
=== Best Results ===
Best Epoch: 8
Training Loss: 0.8765
Entity F1: 85.3%
Triplet F1: 72.1%
```

### Files Created
```
savemodels/dimabsa_combined/best_model.pt
log/dimabsa_combined/train_*.log
log/dimabsa_combined/results_*.json
```

---

## Download Results

### From Kaggle Output Panel (Right Side)
1. Navigate to `savemodels/dimabsa_combined/`
2. Download `best_model.pt` (~1.5 GB)
3. Navigate to `log/dimabsa_combined/`
4. Download training logs and results

---

## Training Time Estimates

| GPU | Time per Epoch | Total (10 epochs) |
|-----|----------------|-------------------|
| T4 x2 | ~15-20 min | ~2.5-3 hours |
| P100 | ~10-12 min | ~1.5-2 hours |
| V100 | ~8-10 min | ~1-1.5 hours |

---

## Troubleshooting

### Out of Memory
**Error**: CUDA out of memory
**Solution**: Change `BATCH_SIZE = 4` to `BATCH_SIZE = 2` in Cell 6

### Clone Failed
**Error**: Repository not found
**Solution**: Check repository is public at https://github.com/VishalRepos/dimabsa-2026

### Data Not Found
**Error**: File not found
**Solution**: Verify you're in `dimabsa-2026/DESS/Codebase` directory

---

## Comparison with Example

| Feature | Example Notebook | Our Notebook |
|---------|-----------------|--------------|
| Repository | DESS-improved | dimabsa-2026 ✅ |
| Dataset | 14res (original) | dimabsa_combined ✅ |
| Task | Classification | VA Regression ✅ |
| Samples | ~1,200 | 3,727 ✅ |
| Output | 3 classes | 2 continuous values ✅ |
| Loss | CrossEntropy | MSE ✅ |
| Epochs | 120 | 10 (faster) ✅ |

---

## Quick Start

1. **Upload**: `kaggle_training_final.ipynb` to Kaggle
2. **Enable**: GPU T4 x2
3. **Run**: Click "Run All"
4. **Wait**: ~2-3 hours
5. **Download**: `best_model.pt`

---

## Files on GitHub

```
✅ kaggle_training_final.ipynb  (NEW - use this!)
✅ kaggle_training.ipynb        (alternative version)
✅ Kaggle_example.ipynb         (reference only)
```

**Use**: `kaggle_training_final.ipynb` for training

---

**Ready to train on Kaggle!** 🚀

*Created: 2026-01-18*
