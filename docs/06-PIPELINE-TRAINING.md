# Pipeline-DeBERTa Training Guide

## Prerequisites

### 1. Data Setup
```bash
# Data should be in DimABSA format (no conversion needed)
ls DimABSA2026/task-dataset/track_a/subtask_2/eng/
# Expected files:
# - eng_restaurant_train_alltasks.jsonl
# - eng_laptop_train_alltasks.jsonl
# - eng_restaurant_dev_task2.jsonl
# - eng_laptop_dev_task2.jsonl
```

### 2. Environment Setup
```bash
cd Pipeline-DeBERTa
pip install -r requirements.txt
```

### 3. Create Symlink (Optional)
```bash
cd Pipeline-DeBERTa
ln -s ../DimABSA2026/task-dataset data
```

---

## Training Configuration

### Command Line Arguments
```python
--task 2                    # Task 2 (triplets) or 3 (quadruplets)
--domain res                # Domain: res, lap, hot, fin
--language eng              # Language: eng, zho, jpn
--bert_model_type microsoft/deberta-v3-base
--mode train                # train or inference
--epoch_num 3               # Number of epochs
--batch_size 8              # Batch size
--learning_rate 1e-3        # LR for classifiers
--tuning_bert_rate 1e-5     # LR for DeBERTa
--inference_beta 0.9        # Confidence threshold
```

---

## Training Commands

### Restaurant Domain
```bash
cd Pipeline-DeBERTa

python run_task2&3_trainer_multilingual.py \
  --task 2 \
  --domain res \
  --language eng \
  --train_data data/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
  --infer_data data/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode train \
  --epoch_num 3 \
  --batch_size 8 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5
```

### Laptop Domain
```bash
python run_task2&3_trainer_multilingual.py \
  --task 2 \
  --domain lap \
  --language eng \
  --train_data data/track_a/subtask_2/eng/eng_laptop_train_alltasks.jsonl \
  --infer_data data/track_a/subtask_2/eng/eng_laptop_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode train \
  --epoch_num 3 \
  --batch_size 8
```

### Combined Training (Both Domains)
```bash
# Concatenate training files first
cat data/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
    data/track_a/subtask_2/eng/eng_laptop_train_alltasks.jsonl \
    > data/track_a/subtask_2/eng/eng_combined_train.jsonl

# Train on combined data
python run_task2&3_trainer_multilingual.py \
  --task 2 \
  --domain combined \
  --language eng \
  --train_data data/track_a/subtask_2/eng/eng_combined_train.jsonl \
  --infer_data data/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode train \
  --epoch_num 3 \
  --batch_size 8
```

---

## Training Process

### Step-by-Step Execution

**Step 1: Data Loading**
```
Loading training data...
✓ Loaded 1,448 samples (restaurant)
Creating queries for 6 steps...
✓ Generated 8,688 queries (1,448 × 6)
```

**Step 2: Training Loop**
```
Epoch 1/3
  Step A (Aspect Forward):
    [====================] 181/181 batches
    Loss: 0.456, Acc: 78.5%
  
  Step OA (Opinion Backward):
    [====================] 181/181 batches
    Loss: 0.523, Acc: 72.3%
  
  Step O (Opinion Forward):
    [====================] 181/181 batches
    Loss: 0.489, Acc: 75.8%
  
  Step AO (Aspect Backward):
    [====================] 181/181 batches
    Loss: 0.512, Acc: 73.2%
  
  Step Valence:
    [====================] 181/181 batches
    Loss: 1.234, RMSE: 1.11
  
  Step Arousal:
    [====================] 181/181 batches
    Loss: 1.456, RMSE: 1.21

Epoch 1 Complete
  Total Loss: 4.670
  Avg Aspect Acc: 75.9%
  Avg Opinion Acc: 74.1%
  Avg VA RMSE: 1.16
```

**Step 3: Model Saving**
```
✓ Model saved to: model/task2_eng_res_epoch1.pth
```

---

## Kaggle Training

### Step 1: Upload Repository
Same as DESS approach (see [03-DESS-TRAINING.md](03-DESS-TRAINING.md))

### Step 2: Kaggle Notebook

**Cell 1 - Setup**:
```python
%cd /kaggle/working
!git clone https://github.com/YOUR-USERNAME/dimabsa-2026.git
%cd dimabsa-2026/Pipeline-DeBERTa

# Verify data
!ls -lh ../DimABSA2026/task-dataset/track_a/subtask_2/eng/
```

**Cell 2 - Install Dependencies**:
```python
!pip install -q transformers==4.36.0 torch==2.1.0
```

**Cell 3 - Train**:
```python
!python run_task2&3_trainer_multilingual.py \
  --task 2 \
  --domain res \
  --language eng \
  --train_data ../DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
  --infer_data ../DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode train \
  --epoch_num 3 \
  --batch_size 8
```

**Cell 4 - Save Model**:
```python
import shutil
shutil.copytree("model", "/kaggle/working/trained_pipeline_model")
!ls -lh /kaggle/working/trained_pipeline_model
```

---

## Training Tips

### Memory Optimization
```bash
# If OOM:
--batch_size 4              # Reduce batch size
--gradient_accumulation 2   # Accumulate gradients
```

### Speed Optimization
```bash
# For faster training:
--epoch_num 2               # Fewer epochs
--batch_size 16             # Larger batch (if memory allows)
```

### Better Performance
```bash
# For better results:
--epoch_num 5               # More epochs
--learning_rate 5e-4        # Lower learning rate
--tuning_bert_rate 5e-6     # Lower DeBERTa LR
--inference_beta 0.85       # Lower threshold (more predictions)
```

---

## Monitoring Training

### Log Files
```bash
# Training logs
cat log/train_task2_eng_res.log

# Tensorboard (if implemented)
tensorboard --logdir=log/
```

### Checkpoints
```bash
# Model checkpoints
ls -lh model/
# Expected files:
# - task2_eng_res_epoch1.pth
# - task2_eng_res_epoch2.pth
# - task2_eng_res_epoch3.pth
# - task2_eng_res_best.pth
```

---

## Troubleshooting

### Issue 1: Data Path Error
```
FileNotFoundError: eng_restaurant_train_alltasks.jsonl not found
```
**Solution**: Check data path or create symlink
```bash
cd Pipeline-DeBERTa
ln -s ../DimABSA2026/task-dataset data
```

### Issue 2: Query Construction Error
```
KeyError: 'Sentence' or 'Text'
```
**Solution**: Check data format (training uses 'Sentence', test uses 'Text')

### Issue 3: VA Loss is NaN
```
Epoch 1: Valence Loss = nan
```
**Solution**:
- Check VA range in data (should be [1, 9])
- Add gradient clipping
- Reduce learning rate

### Issue 4: Low Extraction Accuracy
```
Aspect Accuracy: 45%
```
**Solution**:
- Train for more epochs
- Increase batch size
- Check query templates
- Verify span labels

---

## Model Checkpoints

### Saved Files
```
model/
├── task2_eng_res_epoch1.pth       # Epoch 1 checkpoint
├── task2_eng_res_epoch2.pth       # Epoch 2 checkpoint
├── task2_eng_res_epoch3.pth       # Epoch 3 checkpoint
└── task2_eng_res_best.pth         # Best model
```

### Checkpoint Contents
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'config': {
        'hidden_size': 768,
        'bert_model_type': 'microsoft/deberta-v3-base',
        'num_category': 13
    }
}
```

### Loading Checkpoint
```python
from DimABSAModel import DimABSA
import torch

# Create model
model = DimABSA(hidden_size=768, 
                bert_model_type='microsoft/deberta-v3-base',
                num_category=13)

# Load checkpoint
checkpoint = torch.load('model/task2_eng_res_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## Training Statistics

### Expected Performance
| Metric | Value |
|--------|-------|
| Training Time | 45-60 min (T4 x2, 3 epochs) |
| Epochs | 3 |
| Batch Size | 8 |
| Samples/Epoch | 1,448 (restaurant) |
| Steps/Epoch | ~181 per step × 6 steps |
| Aspect Accuracy | ~75-80% |
| Opinion Accuracy | ~70-75% |
| VA RMSE | ~1.0-1.5 |

### GPU Requirements
| GPU | Batch Size | Time/Epoch |
|-----|------------|------------|
| T4 x2 | 8 | ~15-20 min |
| P100 | 8 | ~10-15 min |
| V100 | 16 | ~5-10 min |

---

## Comparison with DESS Training

| Aspect | Pipeline-DeBERTa | DESS |
|--------|------------------|------|
| **Training Time** | 45-60 min | 30-40 min |
| **Epochs** | 3 | 10 |
| **Batch Size** | 8 | 4 |
| **Steps/Epoch** | ~1,086 (6 steps) | ~932 |
| **Memory** | Higher | Lower |
| **Complexity** | Lower | Higher |

---

## Next Steps

After training completes:
1. Verify model saved: `ls model/`
2. Test inference: See [07-PIPELINE-INFERENCE.md](07-PIPELINE-INFERENCE.md)
3. Generate submission: See [07-PIPELINE-INFERENCE.md](07-PIPELINE-INFERENCE.md)

---

**Previous**: [05-PIPELINE-ARCHITECTURE.md](05-PIPELINE-ARCHITECTURE.md)  
**Next**: [07-PIPELINE-INFERENCE.md](07-PIPELINE-INFERENCE.md)
