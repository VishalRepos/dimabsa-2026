# 🚀 Quick Start: Kaggle Training

## 1️⃣ Upload to Kaggle (2 minutes)

1. Go to https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload: `pipeline-deberta-kaggle.zip`
4. Title: **"Pipeline-DeBERTa Code"**
5. Click **"Create"**

## 2️⃣ Create Notebook (1 minute)

1. Go to https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Settings → Accelerator → **GPU T4** ✅
4. Add Data → Your Datasets → **"Pipeline-DeBERTa Code"**

## 3️⃣ Setup Code (Copy-Paste)

```python
# Extract uploaded code
!unzip -q /kaggle/input/pipeline-deberta-code/Pipeline-DeBERTa.zip -d /kaggle/working/
%cd /kaggle/working/Pipeline-DeBERTa

# Install dependencies
!pip install -q transformers sentencepiece protobuf

# Download dataset
!mkdir -p data/track_a/subtask_2/eng
!wget -q https://raw.githubusercontent.com/DimABSA/DimABSA2026/main/task-dataset/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl -O data/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl
!wget -q https://raw.githubusercontent.com/DimABSA/DimABSA2026/main/task-dataset/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl -O data/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl
!wget -q https://raw.githubusercontent.com/DimABSA/DimABSA2026/main/task-dataset/track_a/subtask_2/eng/eng_laptop_train_alltasks.jsonl -O data/track_a/subtask_2/eng/eng_laptop_train_alltasks.jsonl
!wget -q https://raw.githubusercontent.com/DimABSA/DimABSA2026/main/task-dataset/track_a/subtask_2/eng/eng_laptop_dev_task2.jsonl -O data/track_a/subtask_2/eng/eng_laptop_dev_task2.jsonl
```

## 4️⃣ Train Restaurant (30-45 min)

```python
!python run_task2\\&3_trainer_multilingual.py \
  --task 2 --domain res --language eng \
  --train_data data/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
  --infer_data data/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode train --epoch_num 3 --batch_size 8 \
  --learning_rate 1e-3 --tuning_bert_rate 1e-5 --inference_beta 0.9
```

## 5️⃣ Train Laptop (60-90 min)

```python
!python run_task2\\&3_trainer_multilingual.py \
  --task 2 --domain lap --language eng \
  --train_data data/track_a/subtask_2/eng/eng_laptop_train_alltasks.jsonl \
  --infer_data data/track_a/subtask_2/eng/eng_laptop_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode train --epoch_num 3 --batch_size 8 \
  --learning_rate 1e-3 --tuning_bert_rate 1e-5 --inference_beta 0.9
```

## 6️⃣ Download Results

```python
!zip -r results.zip model/*.pth tasks/subtask_2/*.jsonl log/*.log
```

Then: **Output tab → Download results.zip**

---

## 📊 What to Expect

- **Restaurant F1**: 15-25%
- **Laptop F1**: 12-20%
- **Total Time**: ~2-3 hours
- **Output**: 2 models + 2 prediction files

## 🐛 If Something Goes Wrong

**Out of Memory?**
```python
--batch_size 4  # instead of 8
```

**GPU Not Working?**
- Settings → Accelerator → GPU T4 ✅
- Check: `!nvidia-smi`

**Import Error?**
```python
!pip install -q transformers sentencepiece protobuf
```

---

## ✅ Files Ready

- `pipeline-deberta-kaggle.zip` (29 KB) - Upload this!
- `kaggle_training.ipynb` - Full notebook (optional)
- `KAGGLE_TRAINING_GUIDE.md` - Detailed guide

**You're all set!** 🎉

Upload and train on Kaggle now!
