# Pipeline-DeBERTa Architecture

## Overview

**Approach**: Machine Reading Comprehension (MRC) based pipeline  
**Model**: DeBERTa-v3-base with sequential extraction  
**Original**: Adapted from DimABSA 2026 starter kit

---

## Architecture Diagram

```
Input: "The food was great"
    ↓
┌─────────────────────────────────────────────────────────────┐
│              DeBERTa-v3-base Encoder                        │
│              (Shared across all steps)                      │
└─────────────────────────────────────────────────────────────┘
    ↓
    ├──────────┬──────────┬──────────┬──────────┬──────────┐
    ↓          ↓          ↓          ↓          ↓          ↓
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Step 1  │ │Step 2  │ │Step 3  │ │Step 4  │ │Step 5  │ │Step 6  │
│Aspect  │ │Opinion │ │Opinion │ │Aspect  │ │Valence │ │Arousal │
│Forward │ │Backward│ │Forward │ │Backward│ │        │ │        │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
    ↓          ↓          ↓          ↓          ↓          ↓
  [food]   [great]    [great]    [food]      7.50       7.62
    ↓          ↓          ↓          ↓          ↓          ↓
    └──────────┴──────────┴──────────┴──────────┴──────────┘
                            ↓
                    Combine Results
                            ↓
              ("food", "great", "7.50#7.62")
```

---

## Model Architecture

### Core Model
```python
class DimABSA(nn.Module):
    def __init__(self, hidden_size, bert_model_type, num_category):
        super().__init__()
        
        # Shared encoder
        self.bert = DebertaV2Model.from_pretrained(bert_model_type)
        # microsoft/deberta-v3-base → hidden_size = 768
        
        # Aspect extraction (forward)
        self.classifier_a_start = nn.Linear(hidden_size, 2)
        self.classifier_a_end = nn.Linear(hidden_size, 2)
        
        # Opinion extraction (forward)
        self.classifier_o_start = nn.Linear(hidden_size, 2)
        self.classifier_o_end = nn.Linear(hidden_size, 2)
        
        # Aspect extraction given opinion (backward)
        self.classifier_ao_start = nn.Linear(hidden_size, 2)
        self.classifier_ao_end = nn.Linear(hidden_size, 2)
        
        # Opinion extraction given aspect (backward)
        self.classifier_oa_start = nn.Linear(hidden_size, 2)
        self.classifier_oa_end = nn.Linear(hidden_size, 2)
        
        # Category classification (for Subtask 3)
        self.classifier_category = nn.Linear(hidden_size, num_category)
        
        # VA regression
        self.classifier_valence = nn.Linear(hidden_size, 1)
        self.classifier_arousal = nn.Linear(hidden_size, 1)
```

### Forward Pass
```python
def forward(self, query_tensor, query_mask, query_seg, step):
    # Encode query
    hidden_states = self.bert(
        query_tensor, 
        attention_mask=query_mask, 
        token_type_ids=query_seg
    )[0]  # Shape: (batch, seq_len, 768)
    
    # Route to appropriate classifier
    if step == 'A':  # Aspect forward
        return self.classifier_a_start(hidden_states), \
               self.classifier_a_end(hidden_states)
    
    elif step == 'O':  # Opinion forward
        return self.classifier_o_start(hidden_states), \
               self.classifier_o_end(hidden_states)
    
    elif step == 'AO':  # Aspect given opinion (backward)
        return self.classifier_ao_start(hidden_states), \
               self.classifier_ao_end(hidden_states)
    
    elif step == 'OA':  # Opinion given aspect (backward)
        return self.classifier_oa_start(hidden_states), \
               self.classifier_oa_end(hidden_states)
    
    elif step == 'Valence':
        cls_hidden = hidden_states[:, 0, :]  # [CLS] token
        return self.classifier_valence(cls_hidden).squeeze(-1)
    
    elif step == 'Arousal':
        cls_hidden = hidden_states[:, 0, :]  # [CLS] token
        return self.classifier_arousal(cls_hidden).squeeze(-1)
```

---

## Pipeline Steps

### Step 1: Extract Aspects (Forward)
**Query**: "What is the aspect? [SEP] The food was great"

**Process**:
1. Encode query with DeBERTa
2. Predict start/end positions for aspect spans
3. Extract: "food"

### Step 2: Extract Opinions Given Aspect (Backward)
**Query**: "What is the opinion about food? [SEP] The food was great"

**Process**:
1. Encode query with aspect mention
2. Predict start/end positions for opinion spans
3. Extract: "great"

### Step 3: Extract Opinions (Forward)
**Query**: "What is the opinion? [SEP] The food was great"

**Process**:
1. Encode query
2. Predict opinion spans independently
3. Extract: "great"

### Step 4: Extract Aspects Given Opinion (Backward)
**Query**: "What is the aspect about great? [SEP] The food was great"

**Process**:
1. Encode query with opinion mention
2. Predict aspect spans
3. Extract: "food"

### Step 5: Predict Valence
**Query**: "What is the valence of food great? [SEP] The food was great"

**Process**:
1. Encode query with aspect-opinion pair
2. Use [CLS] token representation
3. Predict valence score: 7.50

### Step 6: Predict Arousal
**Query**: "What is the arousal of food great? [SEP] The food was great"

**Process**:
1. Encode query with aspect-opinion pair
2. Use [CLS] token representation
3. Predict arousal score: 7.62

### Step 7: Combine Results
**Process**:
1. Match aspects from forward/backward extraction
2. Match opinions from forward/backward extraction
3. Pair aspects with opinions
4. Attach VA scores to each pair
5. Filter by confidence threshold

**Output**: `[("food", "great", "7.50#7.62")]`

---

## Key Components

### 1. Query Construction
```python
def make_QA(sentence, aspect=None, opinion=None, step='A'):
    if step == 'A':
        query = f"What is the aspect? {sentence}"
    elif step == 'OA':
        query = f"What is the opinion about {aspect}? {sentence}"
    elif step == 'O':
        query = f"What is the opinion? {sentence}"
    elif step == 'AO':
        query = f"What is the aspect about {opinion}? {sentence}"
    elif step == 'Valence':
        query = f"What is the valence of {aspect} {opinion}? {sentence}"
    elif step == 'Arousal':
        query = f"What is the arousal of {aspect} {opinion}? {sentence}"
    
    return query
```

### 2. Span Extraction
```python
def extract_span(start_logits, end_logits, tokens, threshold=0.5):
    # Get probabilities
    start_probs = torch.softmax(start_logits, dim=-1)[:, :, 1]
    end_probs = torch.softmax(end_logits, dim=-1)[:, :, 1]
    
    # Find valid spans
    spans = []
    for i in range(len(tokens)):
        for j in range(i, min(i+10, len(tokens))):  # Max span length
            if start_probs[i] > threshold and end_probs[j] > threshold:
                score = start_probs[i] * end_probs[j]
                span_text = " ".join(tokens[i:j+1])
                spans.append((span_text, score))
    
    # Return best span
    return max(spans, key=lambda x: x[1]) if spans else (None, 0)
```

### 3. VA Prediction
```python
def predict_va(model, text, aspect, opinion):
    # Valence
    valence_query = f"What is the valence of {aspect} {opinion}? {text}"
    valence_input = tokenizer(valence_query, return_tensors='pt')
    valence_score = model(valence_input, step='Valence')
    
    # Arousal
    arousal_query = f"What is the arousal of {aspect} {opinion}? {text}"
    arousal_input = tokenizer(arousal_query, return_tensors='pt')
    arousal_score = model(arousal_input, step='Arousal')
    
    # Clip to valid range [1, 9]
    valence = torch.clamp(valence_score, 1.0, 9.0)
    arousal = torch.clamp(arousal_score, 1.0, 9.0)
    
    return valence.item(), arousal.item()
```

---

## Loss Functions

### Span Extraction Loss
```python
# Binary cross-entropy for start/end positions
criterion = nn.CrossEntropyLoss()

# For each step (A, O, AO, OA):
start_loss = criterion(start_logits, start_labels)
end_loss = criterion(end_logits, end_labels)
span_loss = start_loss + end_loss
```

### VA Regression Loss
```python
# MSE loss for continuous values
criterion = nn.MSELoss()

valence_loss = criterion(valence_pred, valence_target)
arousal_loss = criterion(arousal_pred, arousal_target)
va_loss = valence_loss + arousal_loss
```

### Total Loss
```python
total_loss = (aspect_loss + opinion_loss + 
              aspect_backward_loss + opinion_backward_loss +
              valence_loss + arousal_loss)
```

---

## Advantages

1. **Simple Architecture**: Just DeBERTa + linear layers
2. **Interpretable**: Each step is explicit
3. **Flexible**: Easy to modify individual steps
4. **No Dependencies**: No need for POS/dependency parsing
5. **Direct Format**: Uses DimABSA format directly

---

## Disadvantages

1. **Multiple Forward Passes**: 6 passes per sample (slower)
2. **Higher Memory**: 6x encoder calls
3. **No Structural Info**: Doesn't use syntax/semantics
4. **Error Propagation**: Errors in early steps affect later steps
5. **Redundancy**: Forward/backward extraction may conflict

---

## Configuration

### Model Parameters
```python
hidden_size = 768  # DeBERTa-v3-base
bert_model_type = "microsoft/deberta-v3-base"
num_category = 13  # For Subtask 3 (not used in Subtask 2)
```

### Training Parameters
```python
batch_size = 8
learning_rate = 1e-3  # For classifiers
tuning_bert_rate = 1e-5  # For DeBERTa
epoch_num = 3
inference_beta = 0.9  # Confidence threshold
```

---

## File Structure

```
Pipeline-DeBERTa/
├── DimABSAModel.py              # Model definition
├── DataProcess.py               # Data loading and query construction
├── Utils.py                     # Utilities (loss, dataset)
├── run_task2&3_trainer_multilingual.py  # Main training script
├── model/                       # Saved checkpoints
├── log/                         # Training logs
└── tasks/subtask_2/            # Predictions
```

---

## Comparison with DESS

| Aspect | Pipeline-DeBERTa | DESS |
|--------|------------------|------|
| **Paradigm** | MRC (query-based) | Span detection |
| **Forward Passes** | 6 per sample | 1 per sample |
| **Complexity** | Low | High |
| **Features** | Text only | Text + POS + Dependency |
| **Memory** | Higher (6x) | Lower (1x) |
| **Speed** | Slower | Faster |
| **Debugging** | Easier | Harder |

---

**Next**: See [06-PIPELINE-TRAINING.md](06-PIPELINE-TRAINING.md) for training instructions
