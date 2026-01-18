# Quick Start: Kaggle Training

## 📦 Files Ready

✅ **Training Notebook**: `kaggle_training.ipynb`  
✅ **Data Package**: `/tmp/dimabsa-dess-data.zip` (820 KB)  
✅ **Setup Guide**: `KAGGLE_SETUP_GUIDE.md`

---

## 🚀 Quick Steps

### 1. Upload Data (5 min)
```
1. Go to: https://www.kaggle.com/datasets
2. Click: "New Dataset"
3. Upload: /tmp/dimabsa-dess-data.zip
4. Title: "DimABSA DESS Training Data"
5. Click: "Create"
```

### 2. Create Notebook (2 min)
```
1. Go to: https://www.kaggle.com/code
2. Click: "New Notebook"
3. Upload: kaggle_training.ipynb
4. Settings → Accelerator → GPU T4 x2
5. Add Data → Your dataset
```

### 3. Update Path (1 min)
```python
# In notebook cell 2, update:
DATA_PATH = "/kaggle/input/YOUR-DATASET-NAME/DESS/Codebase"
```

### 4. Run Training (2-3 hours)
```
Click: "Run All"
Wait: ~2-3 hours
Download: best_model.pt
```

---

## 📊 What to Expect

### Training Progress
```
Epoch 1/10: [████████████] 932/932
  Avg Loss: 2.34
  ✅ Best model saved

Epoch 2/10: [████████████] 932/932
  Avg Loss: 1.82
  ✅ Best model saved
...
```

### Final Output
```
🎉 Training completed!
Best loss: 0.87

Files saved:
✅ /kaggle/working/checkpoints/best_model.pt
✅ /kaggle/working/logs/training_history.json
✅ /kaggle/working/logs/training_curve.png
```

---

## 💾 Download Results

**From Kaggle Output Panel (right side)**:
1. `checkpoints/best_model.pt` (~1.5 GB)
2. `logs/training_history.json`
3. `logs/training_curve.png`

---

## ⚙️ Configuration

| Setting | Value |
|---------|-------|
| Dataset | Combined (3,727 samples) |
| Model | DeBERTa-v3-base |
| Batch Size | 4 |
| Epochs | 10 |
| Learning Rate | 5e-5 |
| GPU | T4 x2 or P100 |
| Time | ~2-3 hours |

---

## 🔧 Troubleshooting

**Out of Memory?**
→ Change `batch_size` from 4 to 2

**Dataset Not Found?**
→ Check path in cell 2

**Slow Training?**
→ Verify GPU is enabled

---

## 📝 Next Steps

After training:
1. Download `best_model.pt`
2. Run inference on test data
3. Generate submission file
4. Submit to competition

---

## 📚 Full Documentation

- **Setup Guide**: `KAGGLE_SETUP_GUIDE.md`
- **Phase 3 Summary**: `PHASE3_KAGGLE_READY.md`
- **Notebook**: `kaggle_training.ipynb`

---

**Ready to train!** 🚀

*Quick start created: 2026-01-18*
