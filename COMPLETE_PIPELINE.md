# Complete Pipeline: Training to Submission

## Visual Flow

```
╔═══════════════════════════════════════════════════════════════════╗
║                         TRAINING PHASE                            ║
╚═══════════════════════════════════════════════════════════════════╝

📁 eng_restaurant_train_alltasks.jsonl
   {"ID": "1", "Text": "sake list was extensive", 
    "Triplet": [{"Aspect": "sake list", "Opinion": "extensive", "VA": "7.83#8.00"}]}
                              ↓
         [convert_dimabsa_to_dess.py]
         • Tokenize: ["sake", "list", "was", "extensive"]
         • Find spans: aspect[0:2], opinion[3:4]
         • Add POS tags (spaCy)
         • Add dependencies (spaCy)
                              ↓
📁 train_dep_triple_polarity_result.json
   {"tokens": ["sake", "list", "was", "extensive"],
    "entities": [{"type": "target", "start": 0, "end": 2}, ...],
    "sentiments": [{"type": "7.83#8.00", "head": 0, "tail": 1}],
    "pos": [...], "dependency": [...]}
                              ↓
              [Modified DESS Model]
              • DeBERTa encoder
              • Dual-channel GCN
              • VA regression head (2 outputs)
              • Train with MSE loss
                              ↓
💾 best_model.pt (Trained checkpoint)


╔═══════════════════════════════════════════════════════════════════╗
║                        INFERENCE PHASE                            ║
╚═══════════════════════════════════════════════════════════════════╝

📁 test_eng_restaurant.jsonl (Competition provides)
   {"ID": "test_1", "Text": "sake list was extensive"}
                              ↓
         [convert_dimabsa_to_dess.py]
         • Same conversion (no labels)
                              ↓
📁 test_dep_triple_polarity_result.json
   {"tokens": ["sake", "list", "was", "extensive"],
    "entities": [], "sentiments": [],  ← Empty (to be predicted)
    "pos": [...], "dependency": [...]}
                              ↓
              [Trained DESS Model]
              • Load best_model.pt
              • Forward pass (evaluate=True)
              • Predict entities + VA scores
                              ↓
🔮 DESS Predictions (Internal format)
   entities: [{"type": "target", "start": 0, "end": 2, "score": 0.95},
              {"type": "opinion", "start": 3, "end": 4, "score": 0.92}]
   sentiments: [{"head": 0, "tail": 1, "va_scores": [7.85, 7.98]}]
                              ↓
         [inference_and_submit.py]
         • Extract entities from predictions
         • Convert token indices → text spans
         • Format VA scores: "7.85#7.98"
                              ↓
📁 pred_eng_restaurant.jsonl (Submission file)
   {"ID": "test_1", 
    "Triplet": [{"Aspect": "sake list", "Opinion": "extensive", "VA": "7.85#7.98"}]}
                              ↓
         [validate_submission.py]
         • Check format
         • Validate VA range [1.00, 9.00]
         • Verify all IDs present
                              ↓
✅ VALID SUBMISSION
                              ↓
         Upload to CodaBench 🚀
```

---

## Key Conversion Points

### Point 1: Training Data Conversion
```
DimABSA (Text + Labels) → DESS (Tokens + Indices + Labels)
```
- **Input**: "sake list was extensive" + VA "7.83#8.00"
- **Output**: tokens[0:2] = aspect, tokens[3:4] = opinion, VA stored
- **Purpose**: Train DESS model

### Point 2: Test Data Conversion
```
DimABSA (Text only) → DESS (Tokens + Indices, no labels)
```
- **Input**: "sake list was extensive"
- **Output**: tokens + linguistic features, empty predictions
- **Purpose**: Prepare for inference

### Point 3: Prediction Conversion
```
DESS (Token indices + VA) → DimABSA (Text spans + VA)
```
- **Input**: entities[0] = tokens[0:2], VA = [7.85, 7.98]
- **Output**: "sake list", "extensive", "7.85#7.98"
- **Purpose**: Create submission file

---

## File Structure

```
DimABSANew/
├── DimABSA2026/
│   └── task-dataset/track_a/subtask_2/eng/
│       ├── eng_restaurant_train_alltasks.jsonl    ← Training input
│       └── eng_restaurant_dev_task2.jsonl         ← Validation input
│
├── DESS/Codebase/
│   ├── data/dimabsa_eng_restaurant/
│   │   ├── train_dep_triple_polarity_result.json  ← Converted training
│   │   └── test_dep_triple_polarity_result.json   ← Converted validation
│   │
│   ├── savemodels/
│   │   └── best_model.pt                          ← Trained checkpoint
│   │
│   └── train.py                                    ← Training script
│
├── scripts/
│   ├── convert_dimabsa_to_dess.py                 ← Forward converter
│   ├── inference_and_submit.py                    ← Inference + reverse
│   └── validate_submission.py                     ← Validation
│
└── submissions/
    └── pred_eng_restaurant.jsonl                  ← Final submission
```

---

## Implementation Checklist

### Phase 1: Data Conversion ✅
- [ ] Implement `convert_dimabsa_to_dess.py`
  - [ ] Tokenization
  - [ ] Span finding
  - [ ] POS tagging (spaCy)
  - [ ] Dependency parsing (spaCy)
  - [ ] JSON output
- [ ] Test on sample data
- [ ] Convert all training datasets

### Phase 2: Model Modification ✅
- [ ] Modify `D2E2S_Model.py`
  - [ ] Replace sentiment classifier with VA regressor
  - [ ] Update forward pass
- [ ] Modify `loss.py`
  - [ ] Add MSE loss for VA
- [ ] Update `Parameter.py`
  - [ ] Add DimABSA dataset configs

### Phase 3: Training ✅
- [ ] Train on English restaurant
- [ ] Validate on dev set
- [ ] Monitor RMSE and continuous F1
- [ ] Save best checkpoint

### Phase 4: Inference & Submission ✅
- [ ] Implement `inference_and_submit.py`
  - [ ] Load trained model
  - [ ] Run predictions
  - [ ] Convert to DimABSA format
  - [ ] Create submission file
- [ ] Implement `validate_submission.py`
- [ ] Test on dev set
- [ ] Generate final submission

### Phase 5: Evaluation ✅
- [ ] Run official evaluation script
- [ ] Compare with baseline
- [ ] Upload to CodaBench

---

## Answer to Your Question

**Q: If we convert DimABSA data to DESS format, how can we submit results?**

**A: We convert BACK from DESS predictions to DimABSA format!**

### The Two-Way Conversion:

1. **Training**: DimABSA → DESS (forward)
2. **Submission**: DESS → DimABSA (reverse)

### Why This Works:

- ✅ We keep original tokens in DESS format
- ✅ Token indices can be mapped back to text
- ✅ VA scores are preserved throughout
- ✅ No information is lost

### The Key Insight:

**DESS is just a processing format, not the final output.**

We use DESS for its powerful model architecture, but we always convert back to DimABSA format for submission. It's like using a different coordinate system for calculations, then converting back to the original system for the final answer.

---

## Next Steps

Ready to implement? The order should be:

1. **Data converter** (forward) - 1 day
2. **Model modifications** - 1 day  
3. **Training** - 1-2 days
4. **Inference converter** (reverse) - 1 day
5. **Submission & validation** - 0.5 day

**Total: ~5 days to first submission!**
