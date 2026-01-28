# Data Format Specifications

## DimABSA Format (Competition Standard)

### Training Data Format
**File**: `eng_restaurant_train_alltasks.jsonl`

```json
{
  "ID": "restaurant_train_001",
  "Sentence": "Great diner food and breakfast is served all day",
  "Triplet": [
    {
      "Aspect": "diner food",
      "Opinion": "Great",
      "VA": "7.50#7.62"
    },
    {
      "Aspect": "breakfast",
      "Opinion": "Great",
      "VA": "7.50#7.62"
    }
  ]
}
```

### Test Data Format
**File**: `eng_restaurant_dev_task2.jsonl`

```json
{
  "ID": "restaurant_test_001",
  "Text": "sake list was extensive"
}
```

### Submission Format
**File**: `pred_eng_restaurant.jsonl`

```json
{
  "ID": "restaurant_test_001",
  "Triplet": [
    {
      "Aspect": "sake list",
      "Opinion": "extensive",
      "VA": "7.85#7.98"
    }
  ]
}
```

---

## DESS Format (Internal Processing)

### Training Data Format
**File**: `train_dep_triple_polarity_result.json`

```json
{
  "tokens": ["Great", "diner", "food", "and", "breakfast", "is", "served", "all", "day"],
  "entities": [
    {"type": "target", "start": 1, "end": 3},
    {"type": "opinion", "start": 0, "end": 1},
    {"type": "target", "start": 4, "end": 5},
    {"type": "opinion", "start": 0, "end": 1}
  ],
  "sentiments": [
    {"type": "7.50#7.62", "head": 0, "tail": 1},
    {"type": "7.50#7.62", "head": 2, "tail": 3}
  ],
  "pos": [
    ["Great", "JJ"],
    ["diner", "NN"],
    ["food", "NN"],
    ...
  ],
  "dependency": [
    [0, 2, "amod"],
    [1, 2, "compound"],
    [2, 6, "nsubjpass"],
    ...
  ]
}
```

**Key Fields**:
- `tokens`: Tokenized text
- `entities`: Aspect (target) and opinion spans by token indices
- `sentiments`: Links between entities with VA scores
  - `head`: Index in entities array (aspect)
  - `tail`: Index in entities array (opinion)
  - `type`: VA string "V.VV#A.AA"
- `pos`: Part-of-speech tags (spaCy format)
- `dependency`: Dependency tree [head_idx, dep_idx, relation]

### Test Data Format
Same structure but `entities` and `sentiments` are empty (to be predicted)

---

## Pipeline-DeBERTa Format (MRC-based)

### Training Data Format
**File**: `eng_restaurant_train_alltasks.jsonl` (same as DimABSA)

Uses the original DimABSA format directly with query-based processing:

**Query Templates**:
```
Step 1 (Aspect): "What is the aspect?"
Step 2 (Opinion): "What is the opinion about [ASPECT]?"
Step 3 (Valence): "What is the valence of [ASPECT] [OPINION]?"
Step 4 (Arousal): "What is the arousal of [ASPECT] [OPINION]?"
```

**Internal Processing**:
```python
# For each sample, create 6 queries:
queries = [
    f"What is the aspect? {text}",
    f"What is the opinion about {aspect}? {text}",
    # ... (forward/backward variations)
    f"What is the valence? {text}",
    f"What is the arousal? {text}"
]
```

---

## Data Conversion

### DimABSA → DESS
**Script**: `scripts/convert_dimabsa_to_dess.py`

```bash
python scripts/convert_dimabsa_to_dess.py \
  --input DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
  --output DESS/Codebase/data/dimabsa_combined/train_dep_triple_polarity_result.json
```

**Process**:
1. Tokenize text (spaCy)
2. Find aspect/opinion spans in tokens
3. Add POS tags
4. Add dependency tree
5. Create entity and sentiment annotations

### DESS → DimABSA
**Script**: `DESS/Codebase/predict.py` (includes conversion)

```python
# Extract predictions
aspect_tokens = tokens[aspect_start:aspect_end]
opinion_tokens = tokens[opinion_start:opinion_end]

# Convert to text
aspect_text = " ".join(aspect_tokens)
opinion_text = " ".join(opinion_tokens)
va_string = f"{valence:.2f}#{arousal:.2f}"

# Create triplet
triplet = {
    "Aspect": aspect_text,
    "Opinion": opinion_text,
    "VA": va_string
}
```

---

## Data Statistics

### Combined Dataset (Restaurant + Laptop)

| Metric | Value |
|--------|-------|
| Training samples | 3,727 |
| Test samples | 400 |
| Total entities | 11,388 |
| Total sentiments | 5,694 |
| Avg entities/sample | 3.06 |
| Avg sentiments/sample | 1.53 |
| Avg tokens/sample | 18.5 |
| Max tokens | 87 |
| Avg Valence | 6.40 |
| Avg Arousal | 7.13 |

### Domain Split

| Domain | Train | Test |
|--------|-------|------|
| Restaurant | 1,448 | ~200 |
| Laptop | 2,279 | ~200 |

---

## Validation Rules

### VA Score Format
- Format: `"V.VV#A.AA"` (string with 2 decimal places)
- Range: `[1.00, 9.00]` for both V and A
- Separator: `#` (no spaces)

### Triplet Requirements
- Aspect: Non-empty string
- Opinion: Non-empty string
- VA: Valid format and range
- All fields required

### Submission Requirements
- All test IDs must be present
- Empty triplet list allowed: `"Triplet": []`
- One JSON object per line (JSONL)
- Valid JSON syntax

---

## File Locations

```
DimABSANew/
├── DimABSA2026/task-dataset/track_a/subtask_2/eng/
│   ├── eng_restaurant_train_alltasks.jsonl    # DimABSA training
│   ├── eng_laptop_train_alltasks.jsonl
│   ├── eng_restaurant_dev_task2.jsonl         # DimABSA test
│   └── eng_laptop_dev_task2.jsonl
│
├── DESS/Codebase/data/dimabsa_combined/
│   ├── train_dep_triple_polarity_result.json  # DESS training
│   └── test_dep_triple_polarity_result.json   # DESS test
│
└── Pipeline-DeBERTa/data/
    └── (symlink to DimABSA2026/task-dataset/)
```
