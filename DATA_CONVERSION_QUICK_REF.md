# Data Conversion: Quick Reference

## Answer to Your Question

**Q: How do we use DimABSA data with DESS?**

**A: We CONVERT DimABSA data to DESS format, then train DESS on the converted data.**

---

## Visual Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DimABSA Format (Input)                       │
├─────────────────────────────────────────────────────────────────────┤
│ {                                                                   │
│   "ID": "rest16_quad_dev_2",                                       │
│   "Text": "their sake list was extensive",                         │
│   "Triplet": [                                                      │
│     {                                                               │
│       "Aspect": "sake list",        ← String (not indices)         │
│       "Opinion": "extensive",       ← String (not indices)         │
│       "VA": "7.83#8.00"            ← Continuous scores             │
│     }                                                               │
│   ]                                                                 │
│ }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    [CONVERSION SCRIPT]
                    - Tokenize text
                    - Find span indices
                    - Generate POS tags (spaCy)
                    - Generate dependencies (spaCy)
                    - Map triplets to entities
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         DESS Format (Output)                        │
├─────────────────────────────────────────────────────────────────────┤
│ {                                                                   │
│   "tokens": ["their", "sake", "list", "was", "extensive"],        │
│   "entities": [                                                     │
│     {"type": "target", "start": 1, "end": 3},  ← Token indices    │
│     {"type": "opinion", "start": 4, "end": 5}  ← Token indices    │
│   ],                                                                │
│   "sentiments": [                                                   │
│     {                                                               │
│       "type": "7.83#8.00",  ← VA score (instead of POS/NEG)       │
│       "head": 0,            ← Index of aspect entity               │
│       "tail": 1             ← Index of opinion entity              │
│     }                                                               │
│   ],                                                                │
│   "pos": [["their","PRP$"], ["sake","NN"], ...],  ← Generated     │
│   "dependency": [["ROOT",0,5], ["poss",3,1], ...], ← Generated    │
│   "orig_id": "rest16_quad_dev_2"                                   │
│ }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                      [TRAIN DESS MODEL]
                      - Use converted data
                      - Model learns VA regression
```

---

## Key Points

### 1. We DON'T Use DESS's Original Data ❌
- DESS comes with SemEval 2014/2015/2016 datasets
- Those have categorical labels (POS/NEG/NEU)
- We IGNORE those completely

### 2. We CONVERT DimABSA Data ✅
- Take DimABSA JSONL files
- Convert to DESS JSON format
- Add linguistic features (POS, dependencies)
- Keep VA scores in sentiment "type" field

### 3. Training Flow
```
DimABSA Data (JSONL)
    ↓
Conversion Script
    ↓
DESS Format (JSON) with VA scores
    ↓
Modified DESS Model (VA regression head)
    ↓
Trained Model for DimABSA
```

---

## File Organization

```
DESS/Codebase/data/
├── 14res/                          ← IGNORE (original DESS data)
├── 14lap/                          ← IGNORE (original DESS data)
├── 15res/                          ← IGNORE (original DESS data)
├── 16res/                          ← IGNORE (original DESS data)
└── dimabsa_eng_restaurant/         ← NEW (converted DimABSA data)
    ├── train_dep_triple_polarity_result.json
    └── test_dep_triple_polarity_result.json
```

---

## Conversion Steps (Simplified)

### Input (DimABSA):
```json
{
  "Text": "their sake list was extensive",
  "Triplet": [{"Aspect": "sake list", "Opinion": "extensive", "VA": "7.83#8.00"}]
}
```

### Step 1: Tokenize
```python
tokens = "their sake list was extensive".split()
# ["their", "sake", "list", "was", "extensive"]
```

### Step 2: Find Spans
```python
aspect = "sake list" → tokens[1:3]  # start=1, end=3
opinion = "extensive" → tokens[4:5]  # start=4, end=5
```

### Step 3: Generate Linguistics
```python
pos_tags = spacy_pos_tagger(tokens)
dependencies = spacy_dependency_parser(tokens)
```

### Step 4: Create DESS Structure
```python
{
  "tokens": ["their", "sake", "list", "was", "extensive"],
  "entities": [
    {"type": "target", "start": 1, "end": 3},
    {"type": "opinion", "start": 4, "end": 5}
  ],
  "sentiments": [
    {"type": "7.83#8.00", "head": 0, "tail": 1}
  ],
  "pos": [...],
  "dependency": [...]
}
```

---

## Why This Works

1. **DESS doesn't care about data source**
   - It just needs the JSON format
   - Linguistic features help the model learn

2. **VA scores fit in "type" field**
   - Originally: "type": "POSITIVE"
   - Modified: "type": "7.83#8.00"
   - Model will learn to predict this

3. **All DESS components work the same**
   - Span extraction: Uses token indices ✅
   - GCN layers: Use dependency trees ✅
   - Sentiment head: Predicts "type" field ✅

---

## Implementation Priority

### Phase 1: Basic Converter (Day 1)
- Tokenization
- Span finding
- Basic POS/dependency generation

### Phase 2: Validation (Day 1)
- Check converted data quality
- Verify span alignments
- Test on small sample

### Phase 3: Full Conversion (Day 2)
- Convert all languages
- Convert all domains
- Create train/dev/test splits

### Phase 4: Integration (Day 2)
- Update DESS Parameter.py
- Point to converted data
- Test data loading

---

## Bottom Line

**You DON'T need DESS's original training data.**

**You CONVERT DimABSA data → DESS format → Train DESS model.**

The conversion script is the bridge between the two formats. Once converted, DESS can train on DimABSA data just like it trained on SemEval data.
