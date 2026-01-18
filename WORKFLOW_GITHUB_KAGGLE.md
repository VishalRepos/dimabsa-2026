# Complete Workflow: GitHub → Kaggle → Training

## Overview

```
Local Development → GitHub → Kaggle → Training → Results
```

---

## Step 1: Push to GitHub (10 min)

### 1.1 Initialize Repository
```bash
cd /Users/vishal.thenuwara/Documents/MSC/Research/Coding/Competition/DimABSANew
bash scripts/init_github.sh
```

### 1.2 Create GitHub Repository
1. Go to https://github.com/new
2. Name: `dimabsa-2026`
3. Description: "DimABSA 2026 - DESS Model with VA Regression"
4. Public or Private
5. **Don't** initialize with README
6. Click "Create"

### 1.3 Push Code
```bash
# Replace YOUR-USERNAME with your GitHub username
git remote add origin https://github.com/YOUR-USERNAME/dimabsa-2026.git
git branch -M main
git push -u origin main
```

**Or use GitHub CLI**:
```bash
gh repo create dimabsa-2026 --public --source=. --remote=origin --push
```

### 1.4 Verify
Check these files are on GitHub:
- ✅ README.md
- ✅ kaggle_training.ipynb
- ✅ DESS/Codebase/ (all files)
- ✅ Data files (8.3 MB + 556 KB)

---

## Step 2: Setup Kaggle (5 min)

### Method A: Use GitHub Integration (Recommended)

1. **Create Kaggle Notebook**
   - Go to https://www.kaggle.com/code
   - Click "New Notebook"
   - Title: "DimABSA Training"

2. **Add GitHub Data**
   - Click "Add Data" → "GitHub"
   - Enter: `YOUR-USERNAME/dimabsa-2026`
   - Kaggle clones your repository

3. **Configure Notebook**
   - Settings → Accelerator → GPU T4 x2
   - Settings → Internet → ON

4. **Upload Training Notebook**
   - File → Upload Notebook
   - Select `kaggle_training.ipynb` from your repo

5. **Update Path**
   ```python
   DATA_PATH = "/kaggle/input/dimabsa-2026/DESS/Codebase"
   ```

### Method B: Use ZIP Upload (Alternative)

1. **Create Package**
   ```bash
   bash scripts/prepare_kaggle_upload.sh
   ```

2. **Upload to Kaggle**
   - Go to https://www.kaggle.com/datasets
   - Create new dataset
   - Upload `/tmp/dimabsa-dess-data.zip`

3. **Create Notebook**
   - Upload `kaggle_training.ipynb`
   - Add dataset to notebook

---

## Step 3: Run Training (2-3 hours)

### 3.1 Start Training
```
Click: "Run All"
```

### 3.2 Monitor Progress
```
Epoch 1/10: [████████████] 932/932
  Avg Loss: 2.34
  ✅ Best model saved

Epoch 2/10: [████████████] 932/932
  Avg Loss: 1.82
  ✅ Best model saved
...
```

### 3.3 Wait for Completion
- T4 GPU: ~2.5-3 hours
- P100 GPU: ~1.5-2 hours

---

## Step 4: Download Results (5 min)

### 4.1 Files to Download
From `/kaggle/working/`:
- ✅ `checkpoints/best_model.pt` (~1.5 GB)
- ✅ `logs/training_history.json`
- ✅ `logs/training_curve.png`

### 4.2 Download Method
- Click on file in Output panel (right side)
- Click "Download"

---

## Step 5: Update GitHub (Optional)

### 5.1 Add Training Results
```bash
# Create results directory
mkdir -p results/

# Add downloaded files
cp ~/Downloads/best_model.pt results/
cp ~/Downloads/training_history.json results/
cp ~/Downloads/training_curve.png results/

# Commit
git add results/
git commit -m "Add training results

- Trained on 3,727 samples
- 10 epochs, best loss: X.XX
- Training time: X hours on Kaggle T4"

git push
```

### 5.2 Create Release
```bash
# Tag release
git tag -a v1.0 -m "First trained model"
git push origin v1.0
```

On GitHub:
1. Go to "Releases"
2. Create new release from tag v1.0
3. Add release notes
4. Attach model file (if < 2GB)

---

## Complete Timeline

| Step | Time | Action |
|------|------|--------|
| 1 | 10 min | Push to GitHub |
| 2 | 5 min | Setup Kaggle |
| 3 | 2-3 hours | Training |
| 4 | 5 min | Download results |
| 5 | 5 min | Update GitHub (optional) |
| **Total** | **~3 hours** | **Complete workflow** |

---

## Workflow Diagram

```
┌─────────────────┐
│ Local Dev       │
│ - Code ready    │
│ - Tests passed  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GitHub          │
│ - Push code     │
│ - Version ctrl  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Kaggle          │
│ - Clone repo    │
│ - Setup GPU     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Training        │
│ - 10 epochs     │
│ - 2-3 hours     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Results         │
│ - Download      │
│ - Update GitHub │
└─────────────────┘
```

---

## Quick Commands

### GitHub
```bash
# Initialize
bash scripts/init_github.sh

# Push
git push

# Tag release
git tag -a v1.0 -m "Trained model"
git push origin v1.0
```

### Kaggle
```python
# In notebook, update path:
DATA_PATH = "/kaggle/input/YOUR-REPO-NAME/DESS/Codebase"
```

---

## Troubleshooting

### GitHub: Large Files
**Error**: File exceeds 100 MB
**Solution**: Data files are only 8.3 MB, should be fine

### Kaggle: Dataset Not Found
**Error**: Path not found
**Solution**: Check DATA_PATH matches your repo name

### Kaggle: Out of Memory
**Error**: CUDA out of memory
**Solution**: Reduce batch_size from 4 to 2

---

## Next Steps After Training

1. **Inference**: Use trained model on test data
2. **Submission**: Generate competition submission file
3. **Evaluation**: Calculate continuous F1 score
4. **Iteration**: Improve model based on results

---

## Files Checklist

Before pushing to GitHub:
- [x] README.md (main documentation)
- [x] .gitignore (exclude unnecessary files)
- [x] requirements.txt (dependencies)
- [x] kaggle_training.ipynb (training notebook)
- [x] DESS/Codebase/ (all model files)
- [x] scripts/ (helper scripts)
- [x] Testing/ (test scripts)
- [x] Documentation (*.md files)

---

**Ready to push to GitHub and train on Kaggle!** 🚀

*Workflow guide created: 2026-01-18*
