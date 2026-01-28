# Vector Extraction - Quick Reference

## 🚀 One-Line Commands

### Extract Training Vectors
```bash
python extract_vectors.py --model_path model/task2_eng_res_best.pth --data_file ../DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl --output_file vectors/train_vectors.jsonl --device cuda
```

### Extract Test Vectors
```bash
python extract_vectors.py --model_path model/task2_eng_res_best.pth --data_file ../DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl --output_file vectors/test_vectors.jsonl --device cuda
```

### Analyze Vectors
```bash
python analyze_vectors.py --vectors_file vectors/train_vectors.jsonl --output_dir analysis/
```

---

## 📊 Load Vectors in Python

```python
from extract_vectors import load_vectors

# Load
data = load_vectors('vectors/train_vectors.jsonl')

# Access
vectors = data['vectors']        # numpy array (N, 768)
ids = data['ids']                # list of IDs
texts = data['texts']            # list of texts
metadata = data['metadata']      # list of dicts
```

---

## 🎯 Common Tasks

### Find Similar Samples
```python
from analyze_vectors import find_similar_samples

query_vec = data['vectors'][0]
similar = find_similar_samples(query_vec, data['vectors'], data['texts'], top_k=5)
```

### Visualize with PCA
```python
from analyze_vectors import visualize_vectors_pca

valences = [m['valence'] for m in data['metadata']]
visualize_vectors_pca(data['vectors'], valences)
```

### Train Classifier
```python
from sklearn.linear_model import Ridge

X = data['vectors']
y = [m['valence'] for m in data['metadata']]

model = Ridge().fit(X, y)
```

---

## 📁 Output Format

```json
{"id": "sample_001", "text": "The food was great", "vector": [0.1, -0.2, ...], "aspect": "food", "opinion": "great", "valence": 7.5, "arousal": 7.6}
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Use `--device cpu` |
| Large files | Compress with `gzip` |
| Missing model | Check `model/` directory |

---

## 📈 File Sizes

- 1,448 samples ≈ 9 MB
- 768 dimensions per vector
- Compressible to ~2 MB

---

**Full docs**: `VECTOR_EXTRACTION.md`
