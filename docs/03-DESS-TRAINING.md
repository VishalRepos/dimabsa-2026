# DESS Training Guide

## Prerequisites

### 1. Data Preparation
```bash
# Data should already be converted to DESS format
ls DESS/Codebase/data/dimabsa_combined/
# Expected files:
# - train_dep_triple_polarity_result.json
# - test_dep_triple_polarity_result.json
# - types_va.json
```

### 2. Environment Setup
```bash
cd DESS/Codebase
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Training Configuration

### Parameter File
**File**: `DESS/Codebase/Parameter.py`

```python
# Dataset configuration
datasets = {
    "dimabsa_combined": {
        "train": "./data/dimabsa_combined/train_dep_triple_polarity_result.json",
        "test": "./data/dimabsa_combined/train_dep_triple_polarity_result.json",
        "types_path": "./data/types_va.json",
    }
}

# Model configuration
pretrained_deberta_name = "microsoft/deberta-v3-base"
emb_dim = 768
deberta_feature_dim = 768
hidden_dim = 384
lstm_layers = 2

# Training configuration
batch_size = 4
epochs = 10
learning_rate = 5e-5
max_span_size = 10
prop_drop = 0.1
```

---

## Local Training

### Basic Command
```bash
cd DESS/Codebase

python train.py \
    --dataset dimabsa_combined \
    --epochs 10 \
    --batch_size 4 \
    --seed 42
```

### Full Command with All Parameters
```bash
python train.py \
    --dataset dimabsa_combined \
    --epochs 10 \
    --batch_size 4 \
    --max_span_size 10 \
    --seed 42 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 \
    --hidden_dim 384 \
    --emb_dim 768 \
    --lstm_layers 2 \
    --learning_rate 5e-5 \
    --prop_drop 0.1
```

### Expected Output
```
Epoch 1/10
  Train Loss: 2.345
  Entity F1: 45.2%
  Sentiment F1: 12.5%
  
Epoch 2/10
  Train Loss: 1.892
  Entity F1: 52.3%
  Sentiment F1: 15.8%
  
...

Epoch 10/10
  Train Loss: 0.456
  Entity F1: 78.5%
  Sentiment F1: 8.22%
  
✓ Best model saved to: savemodels/dimabsa_combined/best_model.pt
```

---

## Kaggle Training (Recommended)

### Step 1: Upload Repository to Kaggle

**Option A: Via GitHub**
1. Push code to GitHub
2. Kaggle Notebook → Add Data → GitHub Repository
3. Select your repository

**Option B: Direct Upload**
1. Create ZIP: `zip -r dimabsa-2026.zip . -x "*.git*" "*.pyc" "__pycache__/*"`
2. Kaggle → Datasets → New Dataset → Upload ZIP
3. Create new notebook and add dataset

### Step 2: Kaggle Notebook Setup

**Cell 1 - Clone and Setup**:
```python
%cd /kaggle/working
!git clone https://github.com/YOUR-USERNAME/dimabsa-2026.git
%cd dimabsa-2026/DESS/Codebase

# Verify data
!ls -lh data/dimabsa_combined/
```

**Cell 2 - Install Dependencies**:
```python
!pip install -q transformers==4.36.0 torch==2.1.0 spacy==3.7.2
!python -m spacy download en_core_web_sm
```

**Cell 3 - Verify Setup**:
```python
import torch
import transformers
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**Cell 4 - Train**:
```python
!python train.py \
    --dataset dimabsa_combined \
    --epochs 10 \
    --batch_size 4 \
    --seed 42
```

**Cell 5 - Save Model**:
```python
import shutil
import os

# Copy trained model to Kaggle output
model_dir = "savemodels/dimabsa_combined"
output_dir = "/kaggle/working/trained_model"

if os.path.exists(model_dir):
    shutil.copytree(model_dir, output_dir)
    print(f"✓ Model saved to {output_dir}")
    !ls -lh {output_dir}
else:
    print("✗ Model directory not found")
```

### Step 3: Download Trained Model
1. Run all cells
2. Wait for training to complete (~30-40 minutes)
3. Check "Output" tab in Kaggle
4. Download `trained_model/` folder

---

## Training Tips

### Memory Optimization
```python
# If OOM (Out of Memory):
--batch_size 2          # Reduce batch size
--max_span_size 8       # Reduce max span size
--gcn_dropout 0.6       # Increase dropout
```

### Speed Optimization
```python
# For faster training:
--epochs 5              # Fewer epochs
--batch_size 8          # Larger batch (if memory allows)
--freeze_transformer    # Freeze DeBERTa (not recommended)
```

### Better Performance
```python
# For better results:
--epochs 15             # More epochs
--learning_rate 3e-5    # Lower learning rate
--prop_drop 0.05        # Less dropout
```

---

## Monitoring Training

### TensorBoard (Optional)
```bash
# In separate terminal
tensorboard --logdir=DESS/Codebase/runs

# Open browser: http://localhost:6006
```

### Log Files
```bash
# Training logs
cat DESS/Codebase/logs/train_dimabsa_combined.log

# Model checkpoints
ls -lh DESS/Codebase/savemodels/dimabsa_combined/
```

---

## Troubleshooting

### Issue 1: Dimension Mismatch
```
RuntimeError: size mismatch, m1: [4 x 768], m2: [1536 x ...]
```
**Solution**: Ensure `deberta_feature_dim=768` and `hidden_dim=384`

### Issue 2: LSTM Hidden State Error
```
RuntimeError: Expected hidden[0] size (2, 4, 384), got (2, 8, 384)
```
**Solution**: Use same `batch_size` as training (4)

### Issue 3: Loss is NaN
```
Epoch 1: Loss = nan
```
**Solution**: 
- Check data format (VA scores should be floats)
- Reduce learning rate: `--learning_rate 1e-5`
- Add gradient clipping (already in code)

### Issue 4: Low F1 Score
```
Sentiment F1: 0.5%
```
**Solution**:
- Check evaluation tolerance (should be ±1.0)
- Verify VA scores are being predicted (not all zeros)
- Train for more epochs

---

## Model Checkpoints

### Saved Files
```
savemodels/dimabsa_combined/
├── best_model.pt              # Best model weights
├── config.json                # Model configuration
├── tokenizer.json             # Tokenizer
├── tokenizer_config.json      # Tokenizer config
└── training_info.json         # Training metadata
```

### Loading Checkpoint
```python
from models.D2E2S_Model import D2E2SModel
import torch

# Load model
model = D2E2SModel(config, cls_token, sentiment_types, entity_types, args)
checkpoint = torch.load("savemodels/dimabsa_combined/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## Training Statistics

### Expected Performance
| Metric | Value |
|--------|-------|
| Training Time | 30-40 min (T4 x2) |
| Epochs | 10 |
| Batch Size | 4 |
| Samples/Epoch | 3,727 |
| Steps/Epoch | ~932 |
| Entity F1 | ~75-80% |
| Sentiment F1 | ~8-10% |

### GPU Requirements
| GPU | Batch Size | Time/Epoch |
|-----|------------|------------|
| T4 x2 | 4 | ~3-4 min |
| P100 | 4 | ~2-3 min |
| V100 | 8 | ~1-2 min |

---

## Next Steps

After training completes:
1. Verify model saved: `ls savemodels/dimabsa_combined/`
2. Test inference: See [04-DESS-INFERENCE.md](04-DESS-INFERENCE.md)
3. Generate submission: See [04-DESS-INFERENCE.md](04-DESS-INFERENCE.md)

---

**Previous**: [02-DESS-ARCHITECTURE.md](02-DESS-ARCHITECTURE.md)  
**Next**: [04-DESS-INFERENCE.md](04-DESS-INFERENCE.md)
