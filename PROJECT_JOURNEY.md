# DimABSA 2026 - Complete Project Journey

## Project Overview
**Competition**: DimABSA 2026 Shared Task - Track A, Subtask 2  
**Task**: Dimensional Aspect Sentiment Triplet Extraction (DimASTE)  
**Goal**: Extract (Aspect, Opinion, VA) triplets with continuous Valence-Arousal scores  
**Model**: DESS (Dual-channel Enhanced Sentiment Span) adapted for VA regression  
**Repository**: https://github.com/VishalRepos/dimabsa-2026.git

---

## Phase 1: Data Preparation ✅

### 1.1 Data Conversion
**Objective**: Convert DimABSA JSONL format to DESS JSON format

**Input Format (DimABSA)**:
```json
{
  "ID": "restaurant_train_001",
  "Sentence": "Great diner food and breakfast is served all day",
  "Triplet": [
    {
      "Aspect": "diner food",
      "Opinion": "Great",
      "VA": "7.50#7.62"
    }
  ]
}
```

**Output Format (DESS)**:
```json
{
  "tokens": ["Great", "diner", "food", "and", "breakfast", "is", "served", "all", "day"],
  "entities": [
    {"type": "target", "start": 1, "end": 3},
    {"type": "opinion", "start": 0, "end": 1}
  ],
  "sentiments": [
    {"type": "7.50#7.62", "head": 0, "tail": 1}
  ],
  "pos": [...],
  "dependency": [...]
}
```

**Script Created**: `scripts/convert_dimabsa_to_dess.py`

**Datasets Converted**:
- Restaurant: 1,448 training samples
- Laptop: 2,279 training samples
- Combined: 3,727 training samples, 400 test samples

**Output Location**: `DESS/Codebase/data/dimabsa_combined/`

**Verification**: All 8/8 tests passed

---

## Phase 2: Model Modifications ✅

### 2.1 Architecture Changes

**File**: `DESS/Codebase/models/D2E2S_Model.py`

**Change 1 - Sentiment Classifier Output**:
```python
# Before: 3-class sentiment classification
self.senti_classifier = nn.Linear(
    config.hidden_size * 3 + self._size_embedding * 2, sentiment_types
)

# After: 2-output VA regression
self.senti_classifier = nn.Linear(
    config.hidden_size * 3 + self._size_embedding * 2, 2  # VA regression
)
```

### 2.2 Loss Function Changes

**File**: `DESS/Codebase/trainer/loss.py`

**Change 1 - Loss Type**:
```python
# Before: Binary Cross Entropy for classification
senti_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

# After: MSE Loss for regression
senti_criterion = torch.nn.MSELoss(reduction="none")
```

**Change 2 - Loss Computation**:
```python
# For VA regression: senti_logits shape [batch, pairs, 2]
senti_logits = senti_logits.view(-1, 2)  # [N, 2]
senti_types = senti_types.view(-1, 2)    # [N, 2]

# MSE loss for VA regression
senti_loss = self._senti_criterion(senti_logits, senti_types)
senti_loss = senti_loss.mean(dim=-1)  # Average over V and A
```

### 2.3 Data Loading Changes

**File**: `DESS/Codebase/trainer/sampling.py`

**Change - VA Scores Instead of One-Hot**:
```python
# Before: One-hot encoding for sentiment classes
senti_types = [r.index for r in pos_senti_types] + neg_senti_types
senti_types_onehot = torch.zeros([senti_types.shape[0], senti_type_count])
senti_types_onehot.scatter_(1, senti_types.unsqueeze(1), 1)

# After: Direct VA scores
senti_types = [r.va_scores if hasattr(r, 'va_scores') else [0.0, 0.0] 
               for r in pos_senti_types] + [[0.0, 0.0]] * len(neg_senti_spans)
senti_types = torch.tensor(senti_types, dtype=torch.float32)  # [N, 2]
```

### 2.4 Input Reader Changes

**File**: `DESS/Codebase/trainer/input_reader.py`

**Change - Parse VA Scores**:
```python
# Parse VA scores from string "V.VV#A.AA"
va_string = jsentiment['type']
if '#' in va_string:
    valence, arousal = map(float, va_string.split('#'))
    sentiment_type = sentimentType(va_string, 1, va_string, va_string, symmetric=False)
    sentiment_type.va_scores = [valence, arousal]
```

---

## Phase 3: Training Setup ✅

### 3.1 Configuration

**File**: `DESS/Codebase/Parameter.py`

**Key Parameters**:
```python
datasets = {
    "dimabsa_combined": {
        "train": "./data/dimabsa_combined/train_dep_triple_polarity_result.json",
        "test": "./data/dimabsa_combined/train_dep_triple_polarity_result.json",  # Use train for eval
        "types_path": "./data/types_va.json",
    }
}

# Model defaults (changed from deberta-v2-xxlarge to deberta-v3-base)
pretrained_deberta_name = "microsoft/deberta-v3-base"
emb_dim = 768  # Changed from 1536
deberta_feature_dim = 768  # Changed from 1536
hidden_dim = 384  # Half of emb_dim for bidirectional LSTM
```

### 3.2 Training Command

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
    --lstm_layers 1
```

### 3.3 Kaggle Training

**Notebook**: `kaggle_training_final.ipynb`

**Steps**:
1. Clone repository
2. Install dependencies
3. Verify data
4. Run training (GPU T4 x2)
5. Model saved to `/kaggle/working/trained_model/`

**Training Results**:
- Epochs: 10
- Batch Size: 4
- Training Time: ~30-40 minutes
- Best F1 Score: ~8.22%

### 3.4 Key Fixes During Training

**Issue 1**: Dimension mismatch (1536 vs 768)
- **Cause**: Hardcoded deberta-v2-xxlarge config
- **Fix**: Load config from `args.pretrained_deberta_name`

**Issue 2**: LSTM dimension mismatch
- **Cause**: Bidirectional LSTM outputs `hidden_dim * 2`
- **Fix**: Set `hidden_dim = 384` (half of 768)

**Issue 3**: Loss function mismatch
- **Cause**: Still using BCEWithLogitsLoss
- **Fix**: Changed to MSELoss

**Issue 4**: Test set has no labels
- **Cause**: Test data is for competition submission
- **Fix**: Use training set for evaluation during training

**Issue 5**: Model not saving
- **Cause**: Missing `_save_best` call with correct parameters
- **Fix**: Added model saving with F1 score tracking

**Issue 6**: Evaluation returns all zeros
- **Cause**: Exact match comparison for continuous VA scores
- **Fix**: Implemented tolerance-based matching (±1.0)

---

## Phase 4: Inference & Submission ✅

### 4.1 Inference Script

**File**: `DESS/Codebase/predict.py`

**Key Features**:
- Loads trained model from checkpoint
- Runs predictions on test data
- Generates submission in DimABSA format
- Handles VA score formatting

**Model Loading**:
```python
# Create args with correct parameters
args = argparse.Namespace(
    size_embedding=25,
    lstm_dim=384,
    hidden_dim=384,
    emb_dim=768,
    deberta_feature_dim=768,
    # ... other parameters
)

# Load model
model = D2E2SModel(config=config, cls_token=cls_token, 
                   sentiment_types=1, entity_types=3, args=args)

# Load weights (supports both .bin and .safetensors)
if os.path.exists(f"{model_path}/model.safetensors"):
    from safetensors.torch import load_file
    state_dict = load_file(f"{model_path}/model.safetensors")
model.load_state_dict(state_dict, strict=False)
```

**Prediction Processing**:
```python
# Get entity predictions
entity_types = entity_clf.argmax(dim=-1)
valid_entities = (entity_types > 0).nonzero()

# Get sentiment predictions
senti_va = senti_clf  # [pairs, 2]
senti_mag = torch.norm(senti_va, dim=-1)
valid_sentis = (senti_mag > 0.1).nonzero()

# Build triplets
for idx in valid_sentis:
    aspect_text = " ".join(tokens[head_span[0]:head_span[1]])
    opinion_text = " ".join(tokens[tail_span[0]:tail_span[1]])
    va_string = f"{valence:.2f}#{arousal:.2f}"
    
    triplets.append({
        "Aspect": aspect_text,
        "Opinion": opinion_text,
        "VA": va_string
    })
```

### 4.2 Kaggle Inference

**Notebook**: `kaggle_inference.ipynb`

**Prerequisites**:
- Upload trained model as Kaggle dataset
- Model folder should contain:
  - `model.safetensors` (791 MB)
  - `config.json`
  - `tokenizer.json`, `tokenizer_config.json`
  - `training_info.json`

**Steps**:

**Cell 1 - Setup**:
```python
%cd /kaggle/working
!rm -rf dimabsa-2026  # Remove old clone
!git clone https://github.com/VishalRepos/dimabsa-2026.git
%cd dimabsa-2026/DESS/Codebase

# Verify test data
!ls -lh data/dimabsa_combined/test_dep_triple_polarity_result.json
```

**Cell 2 - Verify Model**:
```python
import os
model_path = "/kaggle/input/traineddess"  # Update with your dataset name

print(f"Looking for model at: {model_path}")
if os.path.exists(model_path):
    print("\n✓ Model found!")
    !ls -lh {model_path}
```

**Cell 3 - Run Inference**:
```python
!python predict.py \
    --model_path {model_path} \
    --test_data data/dimabsa_combined/test_dep_triple_polarity_result.json \
    --types_path data/types_va.json \
    --output /kaggle/working/submission.json \
    --batch_size 4 \
    --device cuda
```

**Cell 4 - Verify Submission**:
```python
import json
with open('/kaggle/working/submission.json', 'r') as f:
    sub = json.load(f)

print(f"Total samples: {len(sub)}")
print(f"Total triplets: {sum(len(s['Triplet']) for s in sub)}")
print(f"\nFirst prediction:\n{json.dumps(sub[0], indent=2)}")
```

**Cell 5 - Download**:
```python
!ls -lh /kaggle/working/submission.json
# Download from Output panel →
```

### 4.3 Common Issues & Fixes

**Issue 1**: `predict.py` not found
- **Cause**: File not in GitHub repo
- **Fix**: Pushed `predict.py` to repository

**Issue 2**: Model path pointing to file instead of folder
- **Cause**: Used `/kaggle/input/traineddess/model.safetensors`
- **Fix**: Use folder path `/kaggle/input/traineddess`

**Issue 3**: LSTM hidden state size mismatch
- **Cause**: Batch size mismatch (trained with 4, using 8)
- **Fix**: Use `--batch_size 4`

**Issue 4**: KeyError: 'tokens'
- **Cause**: Batch doesn't contain tokens
- **Fix**: Get tokens from `dataset.sentences[sample_idx].tokens`

---

## Submission Format

**Output**: `submission.json`

**Format**:
```json
[
  {
    "ID": "restaurant_test_001",
    "Triplet": [
      {
        "Aspect": "food",
        "Opinion": "great",
        "VA": "7.50#7.62"
      },
      {
        "Aspect": "service",
        "Opinion": "excellent",
        "VA": "8.00#7.80"
      }
    ]
  },
  {
    "ID": "restaurant_test_002",
    "Triplet": []
  }
]
```

**Submission Steps**:
1. Download `submission.json` from Kaggle
2. Go to: https://www.codabench.org/competitions/10918/
3. Click "Submit / View Results"
4. Upload `submission.json`
5. Wait for evaluation (few minutes)
6. Check leaderboard

---

## Repository Structure

```
dimabsa-2026/
├── DESS/
│   └── Codebase/
│       ├── models/              # Model architecture
│       │   ├── D2E2S_Model.py  # Main model (modified for VA)
│       │   ├── Attention_Module.py
│       │   ├── Sem_GCN.py
│       │   └── Syn_GCN.py
│       ├── trainer/             # Training utilities
│       │   ├── loss.py         # MSE loss for VA
│       │   ├── input_reader.py # VA score parsing
│       │   ├── sampling.py     # VA score handling
│       │   └── evaluator.py    # Tolerance-based eval
│       ├── data/                # Converted datasets
│       │   └── dimabsa_combined/
│       │       ├── train_dep_triple_polarity_result.json
│       │       └── test_dep_triple_polarity_result.json
│       ├── train.py            # Training script
│       ├── predict.py          # Inference script
│       └── Parameter.py        # Configuration
├── scripts/
│   └── convert_dimabsa_to_dess.py  # Data converter
├── Testing/                    # Test scripts
├── kaggle_training_final.ipynb # Training notebook
├── kaggle_inference.ipynb      # Inference notebook
├── PHASE1_COMPLETE.md         # Phase 1 documentation
├── PHASE2_COMPLETE.md         # Phase 2 documentation
├── PHASE3_KAGGLE_READY.md     # Phase 3 documentation
├── PHASE4_INFERENCE.md        # Phase 4 documentation
└── README.md                   # Main documentation
```

---

## Key Technical Decisions

### 1. Model Selection
- **Chosen**: DeBERTa-v3-base (768 dimensions)
- **Reason**: Smaller, faster, fits in Kaggle GPU memory
- **Alternative**: DeBERTa-v2-xxlarge (1536 dimensions) - too large

### 2. Loss Function
- **Chosen**: MSE Loss
- **Reason**: Standard for regression tasks
- **Alternative**: Huber Loss, Smooth L1 - not tested

### 3. Evaluation Metric
- **Chosen**: Tolerance-based F1 (±1.0 for VA scores)
- **Reason**: Exact match impossible for continuous values
- **Tolerance**: ±1.0 for both valence and arousal

### 4. Training Strategy
- **Chosen**: Train on combined dataset, evaluate on training set
- **Reason**: Test set has no labels (competition format)
- **Alternative**: Split training set - not implemented

### 5. Batch Size
- **Chosen**: 4
- **Reason**: Fits in GPU memory, LSTM hidden state compatibility
- **Constraint**: LSTM hidden state hardcoded to batch size

---

## Performance Metrics

### Training Performance
- **Dataset**: 3,727 samples
- **Epochs**: 10
- **Training Time**: ~30-40 minutes (Kaggle T4 x2)
- **Best F1**: ~8.22%

### Model Size
- **Model Weights**: 791 MB (model.safetensors)
- **Total Package**: ~800 MB with tokenizer
- **Parameters**: ~184M (DeBERTa-v3-base)

### Inference Performance
- **Test Samples**: 400
- **Batch Size**: 4
- **Inference Time**: ~2-3 minutes (Kaggle GPU)
- **Output Size**: ~50-100 KB (submission.json)

---

## Lessons Learned

### 1. Data Format Matters
- DESS expects specific JSON structure
- POS tags and dependencies required
- VA scores stored as strings "V.VV#A.AA"

### 2. Model Dimensions Must Match
- DeBERTa output → LSTM input → GCN input
- All must use consistent dimensions (768)
- Bidirectional LSTM doubles output size

### 3. Evaluation for Regression ≠ Classification
- Can't use exact match for continuous values
- Need tolerance-based comparison
- F1 score still applicable with proper matching

### 4. Kaggle-Specific Considerations
- GPU memory limits model size
- File size limits for GitHub (100 MB)
- Model must be uploaded as separate dataset

### 5. Debugging Strategy
- Add debug logging at each step
- Verify tensor shapes at each layer
- Test with small batch sizes first

---

## Future Improvements

### 1. Model Architecture
- [ ] Try larger models (DeBERTa-large)
- [ ] Experiment with different LSTM configurations
- [ ] Add attention mechanisms for VA prediction

### 2. Training Strategy
- [ ] Implement proper train/val split
- [ ] Add early stopping
- [ ] Try different learning rates
- [ ] Experiment with batch sizes

### 3. Data Augmentation
- [ ] Synonym replacement
- [ ] Back-translation
- [ ] Paraphrasing

### 4. Evaluation
- [ ] Implement continuous F1 metric
- [ ] Add VA score distance metrics
- [ ] Compare different tolerance values

### 5. Inference Optimization
- [ ] Batch processing optimization
- [ ] Model quantization
- [ ] ONNX export for faster inference

---

## References

- **Competition**: https://www.codabench.org/competitions/10918/
- **Repository**: https://github.com/VishalRepos/dimabsa-2026.git
- **DESS Paper**: [Original DESS paper reference]
- **DeBERTa**: https://huggingface.co/microsoft/deberta-v3-base

---

## Contact & Support

For questions or issues:
1. Open GitHub issue
2. Check documentation files
3. Review error logs in Kaggle

---

**Status**: ✅ Complete - Ready for Submission  
**Last Updated**: 2026-01-19  
**Version**: 1.0
