# Phase 1 Complete: Data Conversion

## Summary
Successfully converted DimABSA data to DESS format for English restaurant dataset.

## Converter Script
**Location**: `scripts/convert_dimabsa_to_dess.py`

**Features**:
- Tokenization using spaCy
- Token span finding for aspects/opinions
- POS tagging and dependency parsing
- VA score preservation (format: "V.VV#A.AA")
- Handles both training (with labels) and test (without labels) data
- Supports both Triplet and Quadruplet fields

## Converted Datasets

### Training Data
- **Input**: `DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl`
- **Output**: `DESS/Codebase/data/dimabsa_eng_restaurant/train_dep_triple_polarity_result.json`
- **Samples**: 1,448 (836 skipped - NULL aspects/opinions)

### Test Data
- **Input**: `DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl`
- **Output**: `DESS/Codebase/data/dimabsa_eng_restaurant/test_dep_triple_polarity_result.json`
- **Samples**: 200 (all converted with empty entities/sentiments)

## Data Format

### Training Sample Structure
```json
{
  "tokens": ["their", "sake", "list", "was", "extensive", ...],
  "entities": [
    {"type": "target", "start": 1, "end": 3},
    {"type": "opinion", "start": 4, "end": 5}
  ],
  "sentiments": [
    {"type": "7.83#8.00", "head": 0, "tail": 1}
  ],
  "pos": [["their", "PRP$"], ["sake", "NN"], ...],
  "dependency": [["poss", 3, 1], ["compound", 3, 2], ...],
  "orig_id": "rest16_quad_dev_2"
}
```

### Test Sample Structure
```json
{
  "tokens": ["Food", "and", "coffee", "are", "great"],
  "entities": [],  // Empty - to be predicted
  "sentiments": [],  // Empty - to be predicted
  "pos": [["Food", "NN"], ["and", "CC"], ...],
  "dependency": [["nsubj", 5, 1], ...],
  "orig_id": "rest26_aste_dev_1"
}
```

## Usage

### Convert Training Data
```bash
python scripts/convert_dimabsa_to_dess.py \
  --input DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
  --output DESS/Codebase/data/dimabsa_eng_restaurant/train_dep_triple_polarity_result.json
```

### Convert Test Data
```bash
python scripts/convert_dimabsa_to_dess.py \
  --input DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
  --output DESS/Codebase/data/dimabsa_eng_restaurant/test_dep_triple_polarity_result.json \
  --test
```

## Key Points

✅ **Lossless Conversion**: Original text preserved via tokens, VA scores stored as strings

✅ **Linguistic Features**: POS tags and dependency parsing added automatically

✅ **Span Mapping**: Text spans converted to token indices (reversible)

✅ **Test Data Support**: Handles unlabeled test data with empty entities/sentiments

## Next Steps

Ready for **Phase 2: Model Modification**
- Modify DESS model to output VA regression instead of sentiment classification
- Update loss function to MSE for VA prediction
- Add continuous F1 evaluation metric
