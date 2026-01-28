# 🚀 Simple Kaggle Setup - Just Upload Notebook!

## ✅ What You Need

1. **File**: `kaggle_pipeline_deberta.ipynb` (just created!)
2. **GitHub**: Push your code to GitHub first
3. **Kaggle**: Account with GPU access

---

## 📋 Step-by-Step Guide

### Step 1: Push Code to GitHub (5 minutes)

```bash
cd /Users/vishal.thenuwara/Documents/MSC/Research/Coding/Competition/DimABSANew

# Initialize git (if not already)
cd Pipeline-DeBERTa
git init
git add .
git commit -m "Pipeline-DeBERTa for DimABSA"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR-USERNAME/dimabsa-pipeline-deberta.git
git branch -M main
git push -u origin main
```

### Step 2: Update Notebook with Your GitHub URL (1 minute)

Open `kaggle_pipeline_deberta.ipynb` and change line in **Step 1**:

```python
# Change this line:
!git clone https://github.com/YOUR-USERNAME/dimabsa-pipeline-deberta.git

# To your actual repo:
!git clone https://github.com/vishal-thenuwara/dimabsa-pipeline-deberta.git
```

### Step 3: Upload to Kaggle (2 minutes)

1. Go to: https://www.kaggle.com/code
2. Click: **"New Notebook"**
3. Click: **"File"** → **"Upload Notebook"**
4. Select: `kaggle_pipeline_deberta.ipynb`
5. Settings → Accelerator → **GPU T4** ✅

### Step 4: Run! (2-3 hours)

Click: **"Run All"** or run cells one by one

That's it! ✅

---

## 🎯 What the Notebook Does

1. **Clones your GitHub repo** (no ZIP needed!)
2. **Installs dependencies** (transformers, sentencepiece, etc.)
3. **Downloads dataset** (from DimABSA GitHub)
4. **Trains Restaurant** (~30-45 min)
5. **Trains Laptop** (~60-90 min)
6. **Validates output** (checks format)
7. **Packages results** (creates ZIP for download)
8. **Shows summary** (F1 scores, statistics)

---

## 📊 Expected Output

```
📊 Restaurant Domain:
  Predictions: 200
  Total triplets: XXX
  Avg triplets/sample: X.XX
  Model: model/task2_res_eng.pth

💻 Laptop Domain:
  Predictions: 200
  Total triplets: XXX
  Avg triplets/sample: X.XX
  Model: model/task2_lap_eng.pth

✅ Training Complete!
📥 Download: pipeline_deberta_results.zip
```

---

## 🐛 Troubleshooting

**Q: Git clone fails?**
```python
# Make sure your repo is public, or use:
!git clone https://YOUR-TOKEN@github.com/YOUR-USERNAME/repo.git
```

**Q: Out of memory?**
```python
# Change batch_size from 8 to 4:
--batch_size 4
```

**Q: GPU not working?**
- Settings → Accelerator → GPU T4 ✅
- Check: `!nvidia-smi`

---

## ✨ Advantages of This Approach

✅ **No ZIP files** - Just upload notebook
✅ **Easy updates** - Push to GitHub, re-run notebook
✅ **Version control** - Git tracks all changes
✅ **Shareable** - Anyone can clone and run
✅ **Clean** - One file to manage

---

## 📁 Files

```
/Users/vishal.thenuwara/Documents/MSC/Research/Coding/Competition/DimABSANew/
└── kaggle_pipeline_deberta.ipynb  ← Upload this to Kaggle!
```

**That's the only file you need!** 🎉

---

## 🎯 Quick Checklist

- [ ] Push Pipeline-DeBERTa to GitHub
- [ ] Update notebook with your GitHub URL
- [ ] Upload notebook to Kaggle
- [ ] Enable GPU T4
- [ ] Click "Run All"
- [ ] Wait 2-3 hours
- [ ] Download results

**Much simpler than ZIP approach!** 🚀
