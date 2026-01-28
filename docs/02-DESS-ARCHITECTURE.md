# DESS Architecture & Modifications

## Overview

**DESS**: Dual-channel Enhanced Sentiment Span  
**Original Task**: ASTE (Aspect Sentiment Triplet Extraction) with 3-class sentiment  
**Adapted For**: DimABSA with continuous VA regression

---

## Architecture Diagram

```
Input Text: "The food was great"
    ↓
[Tokenization + Linguistic Features]
    ↓
tokens: ["The", "food", "was", "great"]
pos: [DT, NN, VBD, JJ]
dependency: [(1,0,"det"), (2,1,"nsubj"), ...]
    ↓
┌─────────────────────────────────────────┐
│  DeBERTa-v3-base Encoder (768-dim)      │
│  Input: token_ids, attention_mask       │
│  Output: contextualized embeddings      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Bi-LSTM (2 layers, hidden=384)         │
│  Captures sequential context            │
└─────────────────────────────────────────┘
    ↓
    ├──────────────────┬──────────────────┐
    ↓                  ↓                  ↓
┌─────────┐    ┌──────────┐    ┌──────────┐
│ Original│    │Syntactic │    │ Semantic │
│ Features│    │  Channel │    │ Channel  │
│    h    │    │          │    │          │
└─────────┘    └──────────┘    └──────────┘
                    ↓                ↓
            ┌──────────────┐  ┌──────────────┐
            │  Syn_GCN     │  │  Sem_GCN     │
            │ (Dependency) │  │ (Similarity) │
            └──────────────┘  └──────────────┘
                    ↓                ↓
            ┌────────────────────────────┐
            │   TIN Fusion Module        │
            │   (Self-Attention)         │
            └────────────────────────────┘
                        ↓
                  Fused Features (h)
                        ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌──────────────────┐        ┌──────────────────┐
│ Entity Classifier│        │Sentiment Classifier│
│ (Span Detection) │        │  (VA Regression)  │
└──────────────────┘        └──────────────────┘
        ↓                               ↓
  Aspect/Opinion                   VA Scores
     Spans                         [V, A]
        ↓                               ↓
        └───────────────┬───────────────┘
                        ↓
              Triplet Formation
                        ↓
    [("food", "great", "7.50#7.62")]
```

---

## Key Components

### 1. Encoder Stack
```python
# DeBERTa encoder
self.deberta = AutoModel.from_pretrained('microsoft/deberta-v3-base')
# Output: (batch, seq_len, 768)

# Bi-LSTM
self.lstm = nn.LSTM(
    input_size=768,
    hidden_size=384,
    num_layers=2,
    bidirectional=True,
    batch_first=True
)
# Output: (batch, seq_len, 768)  # 384*2 from bidirectional
```

### 2. Dual-Channel GCN

**Syntactic Channel** (Dependency-based):
```python
self.Syn_gcn = GCN(emb_dim=768)
# Input: dependency adjacency matrix + features
# Captures grammatical relationships
```

**Semantic Channel** (Similarity-based):
```python
self.Sem_gcn = SemGCN(emb_dim=768)
# Input: semantic similarity graph + features
# Captures meaning relationships
```

### 3. TIN Fusion Module
```python
self.attention_layer = SelfAttention()
# Fuses: original + syntactic + semantic features
# Output: unified representation
```

### 4. Entity Classifier (Span Detection)
```python
self.entity_classifier = nn.Linear(
    hidden_size * 2 + size_embedding,  # 768*2 + 25 = 1561
    entity_types  # 3: [None, Target, Opinion]
)
# Classifies each span as aspect/opinion/neither
```

### 5. Sentiment Classifier (VA Regression) ⚠️ MODIFIED
```python
# ORIGINAL (3-class classification):
self.senti_classifier = nn.Linear(
    config.hidden_size * 3 + size_embedding * 2,  # 768*3 + 25*2 = 2354
    sentiment_types  # 3: [POS, NEG, NEU]
)

# MODIFIED (VA regression):
self.senti_classifier = nn.Linear(
    config.hidden_size * 3 + size_embedding * 2,  # 2354
    2  # [Valence, Arousal]
)
```

**Input Features**:
- Context between aspect-opinion pair (max-pooled)
- Aspect span representation
- Opinion span representation
- Size embeddings for both spans

---

## Modifications for VA Regression

### 1. Model Architecture
**File**: `DESS/Codebase/models/D2E2S_Model.py`

```python
# Line 78-80: Change output dimension
self.senti_classifier = nn.Linear(
    config.hidden_size * 3 + self._size_embedding * 2,
    2  # Changed from sentiment_types (3) to 2 for VA
)
```

### 2. Loss Function
**File**: `DESS/Codebase/trainer/loss.py`

```python
# ORIGINAL:
self._senti_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

# MODIFIED:
self._senti_criterion = torch.nn.MSELoss(reduction="none")

# Loss computation:
senti_loss = self._senti_criterion(senti_logits, senti_types)
senti_loss = senti_loss.mean(dim=-1)  # Average over V and A
```

### 3. Data Loading
**File**: `DESS/Codebase/trainer/sampling.py`

```python
# ORIGINAL: One-hot encoding
senti_types_onehot = torch.zeros([batch_size, sentiment_type_count])
senti_types_onehot.scatter_(1, senti_types.unsqueeze(1), 1)

# MODIFIED: Direct VA scores
senti_types = torch.tensor(
    [[r.va_scores[0], r.va_scores[1]] for r in pos_senti_types],
    dtype=torch.float32
)  # Shape: [N, 2]
```

### 4. Input Reader
**File**: `DESS/Codebase/trainer/input_reader.py`

```python
# Parse VA from string "7.50#7.62"
va_string = jsentiment['type']
if '#' in va_string:
    valence, arousal = map(float, va_string.split('#'))
    sentiment_type.va_scores = [valence, arousal]
```

### 5. Evaluation
**File**: `DESS/Codebase/trainer/evaluator.py`

```python
# ORIGINAL: Exact match
pred_match = (pred_aspect == gold_aspect and 
              pred_opinion == gold_opinion and 
              pred_sentiment == gold_sentiment)

# MODIFIED: Tolerance-based matching
va_match = (abs(pred_valence - gold_valence) <= 1.0 and
            abs(pred_arousal - gold_arousal) <= 1.0)
pred_match = (pred_aspect == gold_aspect and 
              pred_opinion == gold_opinion and 
              va_match)
```

---

## Model Configuration

### DeBERTa-v3-base
```python
pretrained_deberta_name = "microsoft/deberta-v3-base"
emb_dim = 768
deberta_feature_dim = 768
```

### LSTM
```python
hidden_dim = 384  # Half of emb_dim (bidirectional doubles it)
lstm_layers = 2
is_bidirectional = True
```

### GCN
```python
gcn_dim = 768
gcn_dropout = 0.5
```

### Training
```python
batch_size = 4
learning_rate = 5e-5
epochs = 10
max_span_size = 10
prop_drop = 0.1
```

---

## Forward Pass

### Training Mode
```python
def _forward_train(encodings, context_masks, entity_masks, entity_sizes, 
                   entity_spans, entity_types, entity_sample_masks,
                   senti_masks, senti_sizes, senti_spans, senti_types, 
                   senti_sample_masks):
    
    # 1. Encode
    h = self.deberta(encodings, attention_mask=context_masks)[0]
    h, _ = self.lstm(h)
    
    # 2. Dual-channel GCN
    h_syn = self.Syn_gcn(adj_matrix, h)
    h_sem = self.Sem_gcn(h, encodings, seq_lens)
    
    # 3. Fusion
    h = self.attention_layer(h, h_syn, h_sem)
    
    # 4. Entity classification
    entity_clf = self._classify_entities(encodings, h, entity_masks, entity_sizes)
    
    # 5. Sentiment classification (VA regression)
    senti_clf = self._classify_sentiments(encodings, h, senti_masks, senti_sizes, senti_spans)
    
    return entity_clf, senti_clf
```

### Evaluation Mode
```python
def _forward_eval(encodings, context_masks, ...):
    # Same encoding + GCN + fusion
    
    # Entity prediction
    entity_clf = self._classify_entities(...)
    entity_spans = self._filter_spans(entity_clf, ...)
    
    # Generate all possible aspect-opinion pairs
    senti_spans = self._generate_pairs(entity_spans)
    
    # VA prediction for each pair
    senti_clf = self._classify_sentiments(..., senti_spans)
    
    return entity_clf, senti_clf, entity_spans, senti_spans
```

---

## File Structure

```
DESS/Codebase/
├── models/
│   ├── D2E2S_Model.py          # Main model (MODIFIED)
│   ├── Attention_Module.py     # Self-attention
│   ├── Syn_GCN.py              # Syntactic GCN
│   ├── Sem_GCN.py              # Semantic GCN
│   ├── TIN_GCN.py              # Fusion module
│   └── Channel_Fusion.py       # Channel fusion
├── trainer/
│   ├── loss.py                 # Loss functions (MODIFIED)
│   ├── input_reader.py         # Data loading (MODIFIED)
│   ├── sampling.py             # Batch sampling (MODIFIED)
│   ├── evaluator.py            # Evaluation (MODIFIED)
│   └── baseTrainer.py          # Base trainer
├── train.py                    # Training script
├── predict.py                  # Inference script
└── Parameter.py                # Configuration
```

---

## Key Advantages

1. **Single Forward Pass**: All predictions in one pass (vs 6 in Pipeline)
2. **Rich Features**: Syntax + semantics + context
3. **Span-Based**: Natural for entity extraction
4. **Pair Modeling**: Explicit aspect-opinion relationship
5. **Modular**: Easy to modify individual components

---

## Limitations

1. **Complexity**: More components to debug
2. **Dependencies**: Requires POS tags + dependency parsing
3. **Memory**: GCN operations can be memory-intensive
4. **Data Format**: Needs conversion from DimABSA format

---

## Performance

**Training**: ~30-40 minutes (Kaggle T4 x2, 10 epochs)  
**Inference**: ~2-3 minutes (400 samples)  
**Model Size**: 791 MB (model.safetensors)  
**Best F1**: ~8.22% (with tolerance-based matching)

---

**Next**: See [03-DESS-TRAINING.md](03-DESS-TRAINING.md) for training instructions
