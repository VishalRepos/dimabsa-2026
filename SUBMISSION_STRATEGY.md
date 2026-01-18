# Submission Strategy: DESS Predictions → DimABSA Format

## Problem Statement
DESS outputs predictions in its internal format (token indices), but DimABSA competition requires JSONL format with text spans and VA scores.

---

## Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                              │
└─────────────────────────────────────────────────────────────────────┘

DimABSA Training Data (JSONL)
    ↓
[Converter Script] → Convert to DESS format
    ↓
DESS Format (JSON) with VA scores
    ↓
[Modified DESS Model] → Train with VA regression
    ↓
Trained Model Checkpoint

┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PHASE                              │
└─────────────────────────────────────────────────────────────────────┘

Test Data (JSONL - Text only)
    ↓
[Converter Script] → Convert to DESS format (no labels)
    ↓
DESS Format (JSON) - Test set
    ↓
[Trained DESS Model] → Predict entities + VA scores
    ↓
DESS Predictions (token indices + VA)
    ↓
[Reverse Converter] → Convert back to DimABSA format
    ↓
Submission File (JSONL)
    ↓
Upload to CodaBench
```

---

## Test Data Format

### Input (Provided by Competition):
```json
{"ID": "rest16_quad_dev_3", "Text": "the spicy tuna roll was unusually good"}
```

### Required Output (Submission):
```json
{
  "ID": "rest16_quad_dev_3",
  "Triplet": [
    {
      "Aspect": "spicy tuna roll",
      "Opinion": "unusually good",
      "VA": "7.50#7.62"
    }
  ]
}
```

---

## DESS Prediction Format

### What DESS Outputs:
```python
# After model.forward(evaluate=True)
predictions = {
    "tokens": ["the", "spicy", "tuna", "roll", "was", "unusually", "good"],
    "entities": [
        {"type": "target", "start": 1, "end": 4, "score": 0.95},
        {"type": "opinion", "start": 5, "end": 7, "score": 0.92}
    ],
    "sentiments": [
        {
            "head": 0,  # Index of aspect entity
            "tail": 1,  # Index of opinion entity
            "va_scores": [7.50, 7.62],  # [valence, arousal]
            "score": 0.88
        }
    ]
}
```

---

## Reverse Conversion Strategy

### Step 1: Extract Predictions from DESS
```python
def extract_dess_predictions(model_output, tokens):
    """
    Extract triplets from DESS model output
    
    Args:
        model_output: DESS forward pass output
        tokens: Original token list
    
    Returns:
        List of predicted triplets
    """
    entities = model_output['entities']
    sentiments = model_output['sentiments']
    
    triplets = []
    for sentiment in sentiments:
        aspect_idx = sentiment['head']
        opinion_idx = sentiment['tail']
        
        # Get entity spans
        aspect_entity = entities[aspect_idx]
        opinion_entity = entities[opinion_idx]
        
        # Convert token indices to text
        aspect_text = " ".join(tokens[aspect_entity['start']:aspect_entity['end']])
        opinion_text = " ".join(tokens[opinion_entity['start']:opinion_entity['end']])
        
        # Get VA scores
        valence = sentiment['va_scores'][0]
        arousal = sentiment['va_scores'][1]
        va_string = f"{valence:.2f}#{arousal:.2f}"
        
        triplets.append({
            "Aspect": aspect_text,
            "Opinion": opinion_text,
            "VA": va_string
        })
    
    return triplets
```

### Step 2: Format for Submission
```python
def create_submission_file(test_data_path, dess_predictions, output_path):
    """
    Create DimABSA submission file from DESS predictions
    
    Args:
        test_data_path: Path to test JSONL (with IDs and Text)
        dess_predictions: Dictionary mapping ID to predicted triplets
        output_path: Path to save submission JSONL
    """
    submissions = []
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_item = json.loads(line)
            item_id = test_item['ID']
            
            # Get predictions for this ID
            triplets = dess_predictions.get(item_id, [])
            
            # Create submission entry
            submission = {
                "ID": item_id,
                "Triplet": triplets
            }
            submissions.append(submission)
    
    # Write to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for submission in submissions:
            f.write(json.dumps(submission, ensure_ascii=False) + '\n')
    
    print(f"Submission file created: {output_path}")
    print(f"Total predictions: {len(submissions)}")
```

---

## Complete Inference Pipeline

### Script: `inference_and_submit.py`

```python
import json
import torch
from transformers import AutoTokenizer
from models.D2E2S_Model import D2E2SModel
from trainer.input_reader import JsonInputReader

def run_inference_and_create_submission(
    test_jsonl_path,
    model_checkpoint_path,
    output_submission_path
):
    """
    Complete pipeline: Test data → DESS predictions → Submission file
    """
    
    # 1. Load test data and convert to DESS format
    print("Step 1: Converting test data to DESS format...")
    test_dess_path = convert_test_to_dess_format(test_jsonl_path)
    
    # 2. Load trained model
    print("Step 2: Loading trained DESS model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")
    model = D2E2SModel.from_pretrained(model_checkpoint_path)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 3. Run inference
    print("Step 3: Running inference...")
    input_reader = JsonInputReader(...)
    test_dataset = input_reader.get_dataset('test')
    
    all_predictions = {}
    
    for batch in test_dataset:
        with torch.no_grad():
            # Forward pass
            entity_clf, senti_clf, sentiments = model(
                encodings=batch['encodings'],
                context_masks=batch['context_masks'],
                entity_masks=batch['entity_masks'],
                entity_sizes=batch['entity_sizes'],
                entity_spans=batch['entity_spans'],
                entity_sample_masks=batch['entity_sample_masks'],
                adj=batch['adj'],
                evaluate=True
            )
            
            # Extract predictions
            for i, item_id in enumerate(batch['ids']):
                tokens = batch['tokens'][i]
                
                # Get predicted entities
                entities = extract_entities(entity_clf[i], batch['entity_spans'][i])
                
                # Get predicted sentiments with VA scores
                sentiments_pred = extract_sentiments(
                    senti_clf[i], 
                    sentiments[i], 
                    entities
                )
                
                # Convert to triplets
                triplets = convert_to_triplets(entities, sentiments_pred, tokens)
                
                all_predictions[item_id] = triplets
    
    # 4. Create submission file
    print("Step 4: Creating submission file...")
    create_submission_file(test_jsonl_path, all_predictions, output_submission_path)
    
    print(f"✅ Submission ready: {output_submission_path}")

def extract_entities(entity_logits, entity_spans):
    """Extract predicted entities from logits"""
    entity_types = entity_logits.argmax(dim=-1)
    
    entities = []
    for i, (span, type_id) in enumerate(zip(entity_spans, entity_types)):
        if type_id > 0:  # Not "None" type
            entities.append({
                'index': i,
                'type': 'target' if type_id == 1 else 'opinion',
                'start': span[0].item(),
                'end': span[1].item()
            })
    
    return entities

def extract_sentiments(senti_logits, sentiment_pairs, entities):
    """Extract predicted sentiments with VA scores"""
    sentiments = []
    
    for i, (pair, va_scores) in enumerate(zip(sentiment_pairs, senti_logits)):
        head_idx = pair[0].item()
        tail_idx = pair[1].item()
        
        # Check if both entities exist
        if head_idx < len(entities) and tail_idx < len(entities):
            valence = va_scores[0].item()
            arousal = va_scores[1].item()
            
            # Clamp to [1, 9] range
            valence = max(1.0, min(9.0, valence))
            arousal = max(1.0, min(9.0, arousal))
            
            sentiments.append({
                'head': head_idx,
                'tail': tail_idx,
                'valence': valence,
                'arousal': arousal
            })
    
    return sentiments

def convert_to_triplets(entities, sentiments, tokens):
    """Convert entities and sentiments to DimABSA triplet format"""
    triplets = []
    
    for sentiment in sentiments:
        aspect_entity = entities[sentiment['head']]
        opinion_entity = entities[sentiment['tail']]
        
        # Extract text spans
        aspect_text = " ".join(tokens[aspect_entity['start']:aspect_entity['end']])
        opinion_text = " ".join(tokens[opinion_entity['start']:opinion_entity['end']])
        
        # Format VA score
        va_string = f"{sentiment['valence']:.2f}#{sentiment['arousal']:.2f}"
        
        triplets.append({
            "Aspect": aspect_text,
            "Opinion": opinion_text,
            "VA": va_string
        })
    
    return triplets
```

---

## Usage Example

### Training:
```bash
# 1. Convert DimABSA training data to DESS format
python convert_dimabsa_to_dess.py \
    --input DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
    --output DESS/Codebase/data/dimabsa_eng_restaurant/train_dep_triple_polarity_result.json

# 2. Train DESS model
python train.py \
    --dataset dimabsa_eng_restaurant \
    --epochs 10 \
    --batch_size 8
```

### Inference & Submission:
```bash
# 3. Run inference and create submission
python inference_and_submit.py \
    --test_data DimABSA2026/task-dataset/track_a/subtask_2/eng/test_eng_restaurant.jsonl \
    --model_checkpoint DESS/Codebase/savemodels/best_model.pt \
    --output submissions/pred_eng_restaurant.jsonl
```

### Validate Submission:
```bash
# 4. Validate format before uploading
python validate_submission.py \
    --submission submissions/pred_eng_restaurant.jsonl
```

---

## Key Points

### ✅ What We Need to Implement:

1. **Forward Converter** (DimABSA → DESS)
   - For training data: Include labels (VA scores)
   - For test data: Only text, no labels

2. **Reverse Converter** (DESS → DimABSA)
   - Extract predictions from DESS output
   - Convert token indices back to text spans
   - Format VA scores as "V.VV#A.AA"

3. **Inference Pipeline**
   - Load trained model
   - Process test data
   - Generate predictions
   - Create submission file

### ✅ Submission File Requirements:

- **Format**: JSONL (one JSON object per line)
- **Fields**: 
  - `ID`: Must match test data IDs
  - `Triplet`: List of predicted triplets
- **Triplet Fields**:
  - `Aspect`: Text span (string)
  - `Opinion`: Text span (string)
  - `VA`: Format "V.VV#A.AA" (2 decimal places)

### ✅ Validation Checks:

```python
def validate_submission(submission_path):
    """Validate submission file format"""
    with open(submission_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            item = json.loads(line)
            
            # Check required fields
            assert 'ID' in item, f"Line {line_num}: Missing ID"
            assert 'Triplet' in item, f"Line {line_num}: Missing Triplet"
            
            # Check triplet format
            for triplet in item['Triplet']:
                assert 'Aspect' in triplet
                assert 'Opinion' in triplet
                assert 'VA' in triplet
                
                # Validate VA format
                va = triplet['VA']
                assert re.match(r'^\d+\.\d{2}#\d+\.\d{2}$', va), \
                    f"Invalid VA format: {va}"
                
                # Validate VA range [1.00, 9.00]
                v, a = map(float, va.split('#'))
                assert 1.0 <= v <= 9.0, f"Valence out of range: {v}"
                assert 1.0 <= a <= 9.0, f"Arousal out of range: {a}"
    
    print("✅ Submission file is valid!")
```

---

## Summary

### The Complete Flow:

1. **Training**: DimABSA JSONL → DESS JSON → Train Model
2. **Inference**: Test JSONL → DESS JSON → Model Predictions
3. **Submission**: DESS Predictions → DimABSA JSONL → Upload

### Key Scripts to Implement:

1. ✅ `convert_dimabsa_to_dess.py` - Forward conversion
2. ✅ `inference_and_submit.py` - Inference + reverse conversion
3. ✅ `validate_submission.py` - Format validation

### No Data Loss:

- Original text is preserved in DESS format
- Token-to-text mapping is maintained
- Reverse conversion is lossless
- Submission format matches competition requirements

**We have a complete plan from training to submission!** 🎯
