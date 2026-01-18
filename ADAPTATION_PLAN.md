# DESS to DimABSA Adaptation Plan

## Overview
Adapting DESS (Dual-channel Enhanced Sentiment Span) model for DimABSA Track A, Subtask 2: Dimensional Aspect Sentiment Triplet Extraction (DimASTE)

## Key Differences Between DESS and DimABSA

### DESS (Current Model)
- **Task**: Aspect Sentiment Triplet Extraction (ASTE)
- **Output**: (Aspect, Opinion, Sentiment_Polarity)
  - Sentiment: Categorical (POSITIVE, NEGATIVE, NEUTRAL)
- **Data Format**: JSON with entities, sentiments, tokens, POS tags, dependency parsing
- **Example**: 
  ```json
  {
    "entities": [{"type": "target", "start": 4, "end": 5}, {"type": "opinion", "start": 3, "end": 4}],
    "sentiments": [{"type": "NEGATIVE", "head": 0, "tail": 1}],
    "tokens": ["After", "dealing", "with", "subpar", "pizza", ...]
  }
  ```

### DimABSA Subtask 2 (Target)
- **Task**: Dimensional Aspect Sentiment Triplet Extraction (DimASTE)
- **Output**: (Aspect, Opinion, VA_Score)
  - VA Score: Continuous values (Valence#Arousal), range [1.00-9.00], 2 decimal places
- **Data Format**: JSONL with ID, Text, Triplet
- **Example**:
  ```json
  {
    "ID": "rest16_quad_dev_3",
    "Text": "the spicy tuna roll was unusually good...",
    "Triplet": [
      {"Aspect": "spicy tuna roll", "Opinion": "unusually good", "VA": "7.50#7.62"}
    ]
  }
  ```

## Required Modifications

### 1. Data Preprocessing
- **Convert DimABSA JSONL → DESS JSON format**
  - Extract aspect/opinion spans from text
  - Add token indices (start/end positions)
  - Generate POS tags and dependency parsing (using spaCy/stanza)
  - Replace categorical sentiment with VA scores

### 2. Model Architecture Changes
- **Sentiment Classification Head → Regression Head**
  - Current: 3-class classifier (POS/NEG/NEU)
  - Target: 2-output regression (Valence, Arousal)
  - Output range: [1.00, 9.00] with 2 decimal precision
  
- **Loss Function**
  - Current: Cross-Entropy Loss
  - Target: MSE/RMSE Loss for VA regression

### 3. Evaluation Metrics
- **Current**: Precision, Recall, F1 for triplet extraction
- **Target**: Continuous F1 (cF1) metric
  - Categorical match: (Aspect, Opinion) must match exactly
  - VA penalty: Distance-based reduction from perfect match
  - Formula: `cTP = 1 - dist(VA_pred, VA_gold) / D_max`
  - D_max = sqrt(128) for [1,9] scale

### 4. Output Format
- **Post-processing**: Convert model predictions to required JSONL format
  ```json
  {
    "ID": "...",
    "Triplet": [
      {"Aspect": "...", "Opinion": "...", "VA": "V.VV#A.AA"}
    ]
  }
  ```

## Implementation Steps

### Phase 1: Data Conversion
1. Create data converter script: `dimabsa_to_dess_format.py`
   - Parse DimABSA JSONL files
   - Extract aspect/opinion spans using string matching
   - Generate linguistic features (POS, dependency)
   - Store VA scores instead of categorical labels

### Phase 2: Model Modification
1. Modify `models/D2E2S_Model.py`:
   - Replace sentiment classifier with regression head
   - Add VA prediction layer (2 outputs: valence, arousal)
   - Implement value clamping [1.00, 9.00]

2. Update `trainer/loss.py`:
   - Add MSE/RMSE loss for VA regression
   - Combine with span detection loss

3. Modify `trainer/evaluator.py`:
   - Implement continuous F1 metric
   - Add RMSE calculation for VA scores

### Phase 3: Training Pipeline
1. Update `Parameter.py`:
   - Add DimABSA dataset configurations
   - Set paths for converted data

2. Modify `train.py`:
   - Handle VA regression outputs
   - Format predictions correctly

### Phase 4: Inference & Submission
1. Create inference script for test data
2. Format outputs to DimABSA JSONL specification
3. Validate with evaluation script

## File Structure
```
DimABSANew/
├── DESS/                          # Original DESS codebase
│   └── Codebase/
│       ├── models/
│       ├── trainer/
│       ├── data/
│       └── train.py
├── DimABSA2026/                   # Competition data
│   ├── task-dataset/
│   │   └── track_a/subtask_2/
│   └── evaluation_script/
└── adaptation/                    # New adaptation code
    ├── data_converter.py          # Convert DimABSA → DESS format
    ├── modified_model.py          # VA regression model
    ├── continuous_f1.py           # Evaluation metric
    └── inference.py               # Generate submissions
```

## Next Steps
1. Implement data converter for DimABSA → DESS format
2. Modify model architecture for VA regression
3. Test on English restaurant dataset first
4. Expand to other languages/domains
