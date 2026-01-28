# Competition Overview - DimABSA 2026

## Task: Track A - Subtask 2 (DimASTE)

**Dimensional Aspect Sentiment Triplet Extraction**

### Objective
Extract (Aspect, Opinion, VA) triplets from text where:
- **Aspect**: Target entity (e.g., "food", "battery")
- **Opinion**: Sentiment expression (e.g., "great", "terrible")
- **VA**: Valence-Arousal scores in format "V.VV#A.AA" (range: 1.0-9.0)

### Example
```
Input:  "The food was great but service was slow"

Output: [
  {"Aspect": "food", "Opinion": "great", "VA": "7.50#7.62"},
  {"Aspect": "service", "Opinion": "slow", "VA": "3.20#4.15"}
]
```

---

## Valence-Arousal (VA) Space

**Valence**: Negative (1.0) ↔ Positive (9.0)  
**Arousal**: Calm/Sluggish (1.0) ↔ Excited/Energetic (9.0)

Examples:
- "great" → High valence (7.5), High arousal (7.6)
- "terrible" → Low valence (2.0), High arousal (7.0)
- "boring" → Low valence (3.0), Low arousal (2.5)
- "peaceful" → High valence (7.0), Low arousal (3.0)

---

## Dataset

**Domains**: Restaurant + Laptop reviews  
**Languages**: English (primary)

| Split | Samples | Triplets | Avg V | Avg A |
|-------|---------|----------|-------|-------|
| Train | 3,727 | 5,694 | 6.40 | 7.13 |
| Test | 400 | ~600 | - | - |

**Data Location**: `DimABSA2026/task-dataset/track_a/subtask_2/eng/`

---

## Evaluation Metric

**Continuous F1 (cF1)**: Modified F1 that accounts for VA distance

```
D_max = √128  (max distance in [1,9]² space)

For each predicted triplet:
  - Find matching gold triplet (same aspect + opinion)
  - Calculate VA distance: d = √((V_pred - V_gold)² + (A_pred - A_gold)²)
  - Continuous match score: 1 - (d / D_max)

cTP = sum of all continuous match scores
cPrecision = cTP / |predictions|
cRecall = cTP / |gold|
cF1 = 2 × cPrecision × cRecall / (cPrecision + cRecall)
```

**Key**: Exact aspect/opinion match required, VA scores contribute to partial credit

---

## Submission Format

**File**: JSONL (one JSON object per line)

```json
{"ID": "restaurant_test_001", "Triplet": [{"Aspect": "food", "Opinion": "great", "VA": "7.50#7.62"}]}
{"ID": "restaurant_test_002", "Triplet": []}
{"ID": "restaurant_test_003", "Triplet": [{"Aspect": "service", "Opinion": "slow", "VA": "3.20#4.15"}, {"Aspect": "ambiance", "Opinion": "nice", "VA": "6.80#5.50"}]}
```

**Requirements**:
- All test IDs must be present
- VA format: "V.VV#A.AA" (2 decimal places)
- VA range: [1.00, 9.00]
- Empty triplet list if no predictions

---

## Competition Links

- **Codabench**: https://www.codabench.org/competitions/10918/
- **Dataset**: https://github.com/DimABSA/DimABSA2026/tree/main/task-dataset
- **Google Group**: https://groups.google.com/g/dimabsa-participants
- **Discord**: https://discord.gg/xWXDWtkMzu

---

## Important Dates

- **Training Data**: Released
- **Test Data**: Released (no labels)
- **Evaluation**: 20 January 2026
- **Submission Deadline**: TBD

---

## Related Subtasks

**Subtask 1 (DimASR)**: Given aspects, predict VA only  
**Subtask 3 (DimASQP)**: Extract (Aspect, Category, Opinion, VA) quadruplets

*Note: All subtasks use the SAME training data*
