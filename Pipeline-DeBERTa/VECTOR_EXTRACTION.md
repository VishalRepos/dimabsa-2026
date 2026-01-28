# Vector Extraction for Pipeline-DeBERTa

## Overview

Extract [CLS] token embeddings (768-dimensional vectors) from the DeBERTa encoder for each data sample during or after training.

---

## Features

✅ Extract vectors during training or separately  
✅ Save in JSONL format with metadata  
✅ Support for both training and test data  
✅ Includes sample ID, text, vector, and labels (if available)  
✅ Memory-efficient streaming to disk  

---

## Quick Start

### 1. Extract Vectors from Trained Model

```bash
cd Pipeline-DeBERTa

python extract_vectors.py \
  --model_path model/task2_eng_res_best.pth \
  --data_file ../DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
  --output_file vectors/train_vectors.jsonl \
  --device cuda
```

### 2. Extract from Test Data

```bash
python extract_vectors.py \
  --model_path model/task2_eng_res_best.pth \
  --data_file ../DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
  --output_file vectors/test_vectors.jsonl \
  --device cuda
```

---

## Output Format

**JSONL file** (one JSON object per line):

```jsonl
{"id": "restaurant_train_001", "text": "The food was great", "vector": [0.123, -0.456, ...], "aspect": "food", "opinion": "great", "valence": 7.50, "arousal": 7.62}
{"id": "restaurant_train_002", "text": "Service was slow", "vector": [-0.234, 0.567, ...], "aspect": "service", "opinion": "slow", "valence": 3.20, "arousal": 4.15}
```

**Fields**:
- `id`: Sample identifier
- `text`: Original text
- `vector`: 768-dimensional array (DeBERTa-v3-base [CLS] token)
- `aspect`: Aspect term (training data only)
- `opinion`: Opinion term (training data only)
- `valence`: Valence score 1-9 (training data only)
- `arousal`: Arousal score 1-9 (training data only)

---

## Usage in Python

### Load Vectors

```python
from extract_vectors import load_vectors

# Load vectors
data = load_vectors('vectors/train_vectors.jsonl')

print(f"IDs: {data['ids'][:5]}")
print(f"Vectors shape: {data['vectors'].shape}")  # (N, 768)
print(f"Metadata: {data['metadata'][:2]}")
```

### Use for Downstream Tasks

```python
import numpy as np
from sklearn.linear_model import Ridge

# Load training vectors
train_data = load_vectors('vectors/train_vectors.jsonl')
X_train = train_data['vectors']
y_train = np.array([m['valence'] for m in train_data['metadata']])

# Train a simple regressor
model = Ridge()
model.fit(X_train, y_train)

# Load test vectors
test_data = load_vectors('vectors/test_vectors.jsonl')
X_test = test_data['vectors']

# Predict
predictions = model.predict(X_test)
```

---

## Kaggle Training with Vector Extraction

Use the provided notebook: `kaggle_pipeline_deberta.ipynb`

**Steps**:
1. Upload notebook to Kaggle
2. Enable GPU (T4 or P100)
3. Run all cells
4. Download:
   - `trained_pipeline_model/` (model checkpoint)
   - `vectors/train_vectors.jsonl` (training vectors)
   - `vectors/test_vectors.jsonl` (test vectors)

---

## File Sizes

| Dataset | Samples | File Size | Compressed |
|---------|---------|-----------|------------|
| Restaurant Train | 1,448 | ~9 MB | ~2 MB |
| Laptop Train | 2,279 | ~14 MB | ~3 MB |
| Restaurant Test | 200 | ~1.2 MB | ~300 KB |
| Laptop Test | 200 | ~1.2 MB | ~300 KB |

**Compression** (optional):
```bash
gzip vectors/train_vectors.jsonl
# Creates: train_vectors.jsonl.gz
```

---

## Model Modifications

### DimABSAModel.py

Added `return_vectors` parameter to forward pass:

```python
def forward(self, query_tensor, query_mask, query_seg, step, return_vectors=False):
    hidden_states = self.bert(query_tensor, attention_mask=query_mask, 
                               token_type_ids=query_seg)[0]
    
    if return_vectors:
        cls_vector = hidden_states[:, 0, :]  # [batch_size, 768]
        return cls_vector
    
    # Normal forward pass...
```

---

## Use Cases

### 1. Feature Analysis
- Analyze what the model learns
- Visualize embeddings (PCA, t-SNE)
- Cluster similar samples

### 2. Transfer Learning
- Use vectors as features for other tasks
- Train lightweight models on top
- Domain adaptation

### 3. Similarity Search
- Find similar reviews
- Retrieve relevant examples
- Build recommendation systems

### 4. Debugging
- Identify problematic samples
- Analyze model behavior
- Compare different model versions

---

## Advanced Usage

### Extract Vectors During Training

Modify `run_task2&3_trainer_multilingual.py`:

```python
# After each epoch
if epoch % 1 == 0:  # Every epoch
    extract_vectors_from_data(
        model, 
        train_loader, 
        device, 
        f'vectors/epoch_{epoch}_train.jsonl'
    )
```

### Custom Vector Extraction

```python
from extract_vectors import extract_vectors_simple

# Your custom texts
texts = ["The food was amazing", "Service was terrible"]
ids = ["custom_1", "custom_2"]

extract_vectors_simple(
    model, 
    tokenizer, 
    texts, 
    ids, 
    device='cuda',
    output_file='vectors/custom_vectors.jsonl'
)
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory
```bash
# Use CPU
--device cpu

# Or reduce batch size in extract_vectors.py
```

### Issue 2: Large File Size
```bash
# Compress
gzip vectors/*.jsonl

# Or save as float16 (modify extract_vectors.py)
cls_vectors = cls_vectors.half()  # float32 -> float16
```

### Issue 3: Missing Model
```bash
# Check model path
ls -lh model/

# Use correct checkpoint
--model_path model/task2_eng_res_epoch3.pth
```

---

## Performance

| Operation | Time (1,448 samples) | GPU |
|-----------|---------------------|-----|
| Extract vectors | ~2-3 minutes | T4 |
| Save to JSONL | ~5 seconds | - |
| Load from JSONL | ~2 seconds | - |

---

## Next Steps

1. ✅ Extract vectors from trained model
2. ✅ Analyze vector distributions
3. ✅ Use for downstream tasks
4. ✅ Compare with DESS vectors (if needed)

---

**Created**: 2026-01-28  
**Status**: ✅ Ready to use
