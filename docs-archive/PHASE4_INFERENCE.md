# Phase 4: Inference & Submission Guide

## Overview
Generate predictions on test data and create submission file for DimABSA 2026 competition.

## Files Created
- `DESS/Codebase/predict.py` - Inference script
- `kaggle_inference.ipynb` - Kaggle notebook for inference

## Quick Start

### Option 1: Kaggle Notebook (Recommended)

1. **Upload Trained Model to Kaggle**
   - Go to Kaggle → Datasets → New Dataset
   - Upload the `trained_model` folder
   - Name it: `dimabsa-trained-model`

2. **Create New Notebook**
   - Upload `kaggle_inference.ipynb`
   - Add your model dataset as input
   - Run all cells

3. **Download Submission**
   - File will be at `/kaggle/working/submission.json`
   - Download from Output panel

### Option 2: Local Inference

```bash
cd DESS/Codebase

python predict.py \
    --model_path /path/to/trained_model \
    --test_data data/dimabsa_combined/test_dep_triple_polarity_result.json \
    --types_path data/types_va.json \
    --output submission.json \
    --batch_size 8
```

## Output Format

The submission file will be in DimABSA format:

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
  }
]
```

## Submission to Competition

1. **Go to CodaBench**
   - URL: https://www.codabench.org/competitions/10918/
   - Login/Register

2. **Navigate to Submit Tab**
   - Click "Submit / View Results"

3. **Upload Submission**
   - Upload `submission.json`
   - Add description (optional)
   - Click "Submit"

4. **Wait for Results**
   - Evaluation takes a few minutes
   - Check leaderboard for your score

## Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | Required | Path to trained model directory |
| `--test_data` | Required | Path to test JSON file |
| `--types_path` | `data/types_va.json` | Path to entity/sentiment types |
| `--output` | `submission.json` | Output submission file path |
| `--batch_size` | 4 | Batch size for inference |
| `--device` | auto | `cuda` or `cpu` |

## Troubleshooting

### Model Not Found
```bash
# Check model path
ls -lh /path/to/trained_model/

# Should contain:
# - pytorch_model.bin
# - config.json
# - tokenizer files
# - training_info.json
```

### CUDA Out of Memory
```bash
# Reduce batch size
python predict.py --batch_size 2 ...
```

### Wrong Output Format
Check that test data is in DESS format with:
- `tokens`: list of words
- `entities`: empty list (unlabeled)
- `sentiments`: empty list (unlabeled)
- `pos`: POS tags
- `dependency`: dependency parse

## Expected Performance

Based on training F1 score of ~8%, expect:
- **Precision**: ~8-10%
- **Recall**: ~8-10%
- **F1 Score**: ~8-10%

This is a baseline. Improvements possible through:
- More training epochs
- Hyperparameter tuning
- Data augmentation
- Ensemble methods

## Next Steps

After submission:
1. Check competition leaderboard
2. Analyze errors on validation set
3. Iterate on model improvements
4. Re-train and re-submit

## Competition Details

- **Task**: Dimensional Aspect Sentiment Triplet Extraction (DimASTE)
- **Track**: Track A - Subtask 2
- **Metric**: Continuous F1 (with tolerance for VA scores)
- **Deadline**: Check competition page

---

**Status**: ✅ Ready for Inference
**Last Updated**: 2026-01-18
