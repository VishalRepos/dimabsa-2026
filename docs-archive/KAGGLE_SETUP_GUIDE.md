# Kaggle Training Setup Guide

## Step 1: Prepare Files for Upload

### Create a ZIP file with the following structure:

```
dimabsa-dess-data.zip
└── DESS/
    └── Codebase/
        ├── models/
        │   ├── D2E2S_Model.py
        │   ├── Syn_GCN.py
        │   ├── Sem_GCN.py
        │   ├── Attention_Module.py
        │   ├── Channel_Fusion.py
        │   ├── TIN_GCN.py
        │   └── General.py
        ├── trainer/
        │   ├── loss.py
        │   ├── input_reader.py
        │   ├── entities.py
        │   ├── util.py
        │   ├── sampling.py
        │   └── evaluator.py
        ├── data/
        │   ├── dimabsa_combined/
        │   │   ├── train_dep_triple_polarity_result.json
        │   │   └── test_dep_triple_polarity_result.json
        │   └── types_va.json
        ├── Parameter.py
        └── train.py
```

### Commands to create the ZIP:

```bash
cd /Users/vishal.thenuwara/Documents/MSC/Research/Coding/Competition/DimABSANew

# Create temporary directory
mkdir -p /tmp/dimabsa-upload/DESS/Codebase

# Copy models
cp -r DESS/Codebase/models /tmp/dimabsa-upload/DESS/Codebase/

# Copy trainer
cp -r DESS/Codebase/trainer /tmp/dimabsa-upload/DESS/Codebase/

# Copy data
mkdir -p /tmp/dimabsa-upload/DESS/Codebase/data/dimabsa_combined
cp DESS/Codebase/data/dimabsa_combined/*.json /tmp/dimabsa-upload/DESS/Codebase/data/dimabsa_combined/
cp DESS/Codebase/data/types_va.json /tmp/dimabsa-upload/DESS/Codebase/data/

# Copy config files
cp DESS/Codebase/Parameter.py /tmp/dimabsa-upload/DESS/Codebase/
cp DESS/Codebase/train.py /tmp/dimabsa-upload/DESS/Codebase/

# Create ZIP
cd /tmp/dimabsa-upload
zip -r dimabsa-dess-data.zip DESS/

echo "ZIP file created: /tmp/dimabsa-upload/dimabsa-dess-data.zip"
```

---

## Step 2: Upload to Kaggle

### 2.1 Create New Dataset

1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload `dimabsa-dess-data.zip`
4. Title: "DimABSA DESS Training Data"
5. Make it Private
6. Click "Create"

### 2.2 Note the Dataset Path

After upload, note the dataset path (e.g., `your-username/dimabsa-dess-data`)

---

## Step 3: Create Kaggle Notebook

### 3.1 Create New Notebook

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Title: "DimABSA Training - DESS VA Regression"

### 3.2 Configure Notebook

**Settings**:
- Accelerator: GPU T4 x2 (or P100)
- Internet: ON (for downloading models)
- Persistence: Files only

**Add Dataset**:
- Click "Add Data" → "Your Datasets"
- Select "DimABSA DESS Training Data"

### 3.3 Upload Notebook

1. Download `kaggle_training.ipynb` from this project
2. In Kaggle notebook, click "File" → "Upload Notebook"
3. Select `kaggle_training.ipynb`

---

## Step 4: Update Paths in Notebook

In the notebook, update the `DATA_PATH` variable:

```python
# Change this line based on your dataset path
DATA_PATH = "/kaggle/input/dimabsa-dess-data/DESS/Codebase"
```

---

## Step 5: Run Training

1. Click "Run All" or run cells sequentially
2. Monitor training progress
3. Training will take approximately:
   - With T4 GPU: ~2-3 hours
   - With P100 GPU: ~1-2 hours

---

## Step 6: Download Results

After training completes:

1. **Trained Model**: `/kaggle/working/checkpoints/best_model.pt`
2. **Training History**: `/kaggle/working/logs/training_history.json`
3. **Training Curve**: `/kaggle/working/logs/training_curve.png`

Download these files from the Kaggle output panel (right side).

---

## Expected Output

### Training Progress
```
Epoch 1/10: 100%|██████████| 932/932 [15:23<00:00]
  Avg Loss: 2.3456
  Time: 923.45s
  ✅ Best model saved (loss: 2.3456)

Epoch 2/10: 100%|██████████| 932/932 [15:20<00:00]
  Avg Loss: 1.8234
  Time: 920.12s
  ✅ Best model saved (loss: 1.8234)
...
```

### Final Output
```
🎉 Training completed!
Best loss: 0.8765
```

---

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` from 4 to 2
- Use smaller model: `deberta-v3-small` instead of `deberta-v3-base`

### Dataset Not Found
- Check dataset path in notebook
- Ensure dataset is added to notebook
- Verify ZIP structure

### Slow Training
- Ensure GPU is enabled (Settings → Accelerator)
- Check GPU utilization: `!nvidia-smi`

---

## Next Steps After Training

1. Download trained model (`best_model.pt`)
2. Use for inference on test data
3. Generate submission file for DimABSA competition
4. Evaluate using official metrics

---

## File Sizes (Approximate)

- ZIP file: ~50 MB
- Training data: ~10 MB
- Trained model: ~1.5 GB (deberta-v3-base)
- Kaggle storage: ~2 GB total

---

## Alternative: Google Colab

If Kaggle doesn't work, you can also use Google Colab:

1. Upload ZIP to Google Drive
2. Mount Drive in Colab
3. Run similar notebook with adjusted paths

---

*Setup guide created: 2026-01-18*
