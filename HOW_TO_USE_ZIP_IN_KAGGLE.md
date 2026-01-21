# How to Use ZIP File in Kaggle - Step by Step

## 📋 Method 1: Upload as Dataset (Recommended)

### Step 1: Upload ZIP to Kaggle Datasets

1. **Go to Kaggle Datasets**
   - Visit: https://www.kaggle.com/datasets
   - Click: **"New Dataset"** button (top right)

2. **Upload the ZIP file**
   - Click: **"Upload"** or drag-and-drop
   - Select: `pipeline-deberta-kaggle.zip`
   - Wait for upload to complete

3. **Fill in Details**
   - **Title**: `Pipeline-DeBERTa Code`
   - **Subtitle**: (optional) `DimABSA training code with DeBERTa`
   - **Description**: (optional) `Code for DimABSA Subtask 2`
   - **Visibility**: **Private** (recommended)

4. **Create Dataset**
   - Click: **"Create"** button
   - Wait for processing (~30 seconds)
   - Note the dataset name (e.g., `yourusername/pipeline-deberta-code`)

### Step 2: Create Kaggle Notebook

1. **Go to Kaggle Notebooks**
   - Visit: https://www.kaggle.com/code
   - Click: **"New Notebook"** button

2. **Enable GPU**
   - Click: **Settings** (right sidebar)
   - **Accelerator**: Select **"GPU T4"** or **"GPU P100"**
   - Click: **"Save"**

3. **Add Your Dataset**
   - Click: **"Add Data"** (right sidebar)
   - Click: **"Your Datasets"** tab
   - Find: `Pipeline-DeBERTa Code`
   - Click: **"Add"** button

### Step 3: Extract ZIP in Notebook

**In the first code cell, paste this:**

```python
# Extract the uploaded code
!unzip -q /kaggle/input/pipeline-deberta-code/pipeline-deberta-kaggle.zip -d /kaggle/working/

# Change to the code directory
%cd /kaggle/working/Pipeline-DeBERTa

# Verify extraction
!ls -la
```

**Run the cell** - You should see all the Python files listed!

### Step 4: Continue with Training

Now follow the rest of the training steps from `QUICK_START_KAGGLE.md`

---

## 📋 Method 2: Direct Upload to Notebook (Simpler but Limited)

### Step 1: Create Notebook
- Go to: https://www.kaggle.com/code
- Click: **"New Notebook"**
- Enable: **GPU T4**

### Step 2: Upload ZIP Directly

**In a code cell:**

```python
# Upload file (this will show a file picker)
from google.colab import files
uploaded = files.upload()  # Select pipeline-deberta-kaggle.zip
```

**Note**: This only works in Colab, not Kaggle!

**For Kaggle, use Method 1 (Dataset upload)**

---

## 🎯 Quick Visual Guide

```
┌─────────────────────────────────────────┐
│  1. Upload ZIP to Kaggle Datasets       │
│     https://kaggle.com/datasets         │
│     → New Dataset → Upload ZIP          │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  2. Create Notebook with GPU            │
│     https://kaggle.com/code             │
│     → New Notebook → GPU T4             │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  3. Add Your Dataset to Notebook        │
│     Add Data → Your Datasets            │
│     → Pipeline-DeBERTa Code → Add       │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  4. Extract ZIP in First Cell           │
│     !unzip -q /kaggle/input/.../zip     │
│     %cd /kaggle/working/Pipeline-DeBERTa│
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  5. Run Training Commands               │
│     (See QUICK_START_KAGGLE.md)         │
└─────────────────────────────────────────┘
```

---

## 📝 Complete First Cell Example

Copy-paste this into your first Kaggle notebook cell:

```python
# ========================================
# SETUP: Extract Code and Install Dependencies
# ========================================

# 1. Extract uploaded code
print("Extracting code...")
!unzip -q /kaggle/input/pipeline-deberta-code/pipeline-deberta-kaggle.zip -d /kaggle/working/
%cd /kaggle/working/Pipeline-DeBERTa

# 2. Install dependencies
print("\nInstalling dependencies...")
!pip install -q transformers sentencepiece protobuf

# 3. Download dataset
print("\nDownloading dataset...")
!mkdir -p data/track_a/subtask_2/eng

!wget -q https://raw.githubusercontent.com/DimABSA/DimABSA2026/main/task-dataset/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
    -O data/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl

!wget -q https://raw.githubusercontent.com/DimABSA/DimABSA2026/main/task-dataset/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
    -O data/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl

!wget -q https://raw.githubusercontent.com/DimABSA/DimABSA2026/main/task-dataset/track_a/subtask_2/eng/eng_laptop_train_alltasks.jsonl \
    -O data/track_a/subtask_2/eng/eng_laptop_train_alltasks.jsonl

!wget -q https://raw.githubusercontent.com/DimABSA/DimABSA2026/main/task-dataset/track_a/subtask_2/eng/eng_laptop_dev_task2.jsonl \
    -O data/track_a/subtask_2/eng/eng_laptop_dev_task2.jsonl

# 4. Verify setup
print("\n✓ Setup complete!")
print("\nFiles extracted:")
!ls -la

print("\nDataset downloaded:")
!ls -lh data/track_a/subtask_2/eng/

# 5. Check GPU
import torch
print(f"\nGPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## ❓ Common Questions

**Q: Where does the ZIP get extracted?**
A: To `/kaggle/working/Pipeline-DeBERTa/`

**Q: Can I see the files after extraction?**
A: Yes! Run `!ls -la` in a cell

**Q: What if the path is different?**
A: Check your dataset name in Kaggle. It might be:
- `/kaggle/input/pipeline-deberta-code/pipeline-deberta-kaggle.zip`
- `/kaggle/input/your-dataset-name/pipeline-deberta-kaggle.zip`

**Q: Do I need to upload the ZIP every time?**
A: No! Once uploaded as a dataset, just add it to any notebook

**Q: Can I modify the code after extraction?**
A: Yes! Edit files in `/kaggle/working/Pipeline-DeBERTa/`

---

## ✅ Verification

After running the first cell, you should see:

```
Extracting code...
Installing dependencies...
Downloading dataset...
✓ Setup complete!

Files extracted:
DimABSAModel.py
Utils.py
DataProcess.py
run_task2&3_trainer_multilingual.py
...

Dataset downloaded:
eng_restaurant_train_alltasks.jsonl (XXX KB)
eng_restaurant_dev_task2.jsonl (XXX KB)
eng_laptop_train_alltasks.jsonl (XXX KB)
eng_laptop_dev_task2.jsonl (XXX KB)

GPU available: True
GPU: Tesla T4
```

If you see this, you're ready to train! 🚀

---

## 🎯 Next Step

After setup cell runs successfully, add a second cell with the training command:

```python
# Train Restaurant Domain
!python run_task2\&3_trainer_multilingual.py \
  --task 2 --domain res --language eng \
  --train_data data/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
  --infer_data data/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode train --epoch_num 3 --batch_size 8 \
  --learning_rate 1e-3 --tuning_bert_rate 1e-5 --inference_beta 0.9
```

That's it! 🎉
