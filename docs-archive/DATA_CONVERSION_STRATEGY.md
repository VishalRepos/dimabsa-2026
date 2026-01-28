# Data Conversion Strategy: DimABSA → DESS Format

## Problem Statement
DimABSA provides data in JSONL format with text and span strings, while DESS requires JSON format with token indices, POS tags, and dependency parsing.

---

## Format Comparison

### DimABSA Format (Input)
```json
{
  "ID": "rest16_quad_dev_2",
  "Text": "their sake list was extensive , but we were looking for purple haze , which was n ' t listed but made for us upon request !",
  "Quadruplet": [
    {
      "Aspect": "sake list",
      "Opinion": "extensive",
      "Category": "DRINKS#STYLE_OPTIONS",
      "VA": "7.83#8.00"
    }
  ]
}
```

**Key Characteristics:**
- ✅ Plain text with aspect/opinion as strings
- ✅ VA scores in "V#A" format
- ❌ No token indices
- ❌ No POS tags
- ❌ No dependency parsing
- ❌ No tokenization

### DESS Format (Required)
```json
{
  "tokens": ["their", "sake", "list", "was", "extensive", ",", ...],
  "entities": [
    {"type": "target", "start": 1, "end": 3},    // "sake list" = tokens[1:3]
    {"type": "opinion", "start": 4, "end": 5}    // "extensive" = tokens[4:5]
  ],
  "sentiments": [
    {
      "type": "7.83#8.00",  // VA score instead of "POSITIVE"
      "head": 0,            // Index of aspect entity
      "tail": 1             // Index of opinion entity
    }
  ],
  "pos": [["their", "PRP$"], ["sake", "NN"], ["list", "NN"], ...],
  "dependency": [["ROOT", 0, 5], ["poss", 3, 1], ...],
  "orig_id": "rest16_quad_dev_2"
}
```

**Key Characteristics:**
- ✅ Tokenized text
- ✅ Token-level indices for spans
- ✅ POS tags for each token
- ✅ Dependency parse tree
- ✅ Entity-sentiment relationships

---

## Conversion Pipeline

### Step 1: Tokenization
**Challenge**: DimABSA text is pre-tokenized (spaces between tokens), but we need to align with DeBERTa tokenizer.

**Solution**: Use the SAME tokenizer as DESS (DeBERTa-v2-xxlarge)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")

# DimABSA text (already space-separated)
text = "their sake list was extensive , but we were looking for purple haze"

# Tokenize
tokens = text.split()  # Use simple split since DimABSA is pre-tokenized
# Result: ["their", "sake", "list", "was", "extensive", ",", ...]
```

### Step 2: Find Span Indices
**Challenge**: Convert aspect/opinion strings to token indices

**Solution**: String matching with tokenized text

```python
def find_span_indices(tokens, span_text):
    """
    Find start and end indices of span_text in tokens list
    
    Args:
        tokens: ["their", "sake", "list", "was", "extensive"]
        span_text: "sake list"
    
    Returns:
        (start, end): (1, 3)  # tokens[1:3] = ["sake", "list"]
    """
    if span_text == "NULL":
        return None
    
    span_tokens = span_text.split()
    span_len = len(span_tokens)
    
    for i in range(len(tokens) - span_len + 1):
        if tokens[i:i+span_len] == span_tokens:
            return (i, i + span_len)
    
    # Fallback: fuzzy matching for tokenization mismatches
    return fuzzy_find_span(tokens, span_tokens)
```

### Step 3: POS Tagging
**Challenge**: Generate POS tags for each token

**Solution**: Use spaCy or NLTK

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def get_pos_tags(tokens):
    """
    Generate POS tags for tokens
    
    Args:
        tokens: ["their", "sake", "list", "was", "extensive"]
    
    Returns:
        [["their", "PRP$"], ["sake", "NN"], ["list", "NN"], ...]
    """
    # Reconstruct text for spaCy
    text = " ".join(tokens)
    doc = nlp(text)
    
    pos_tags = []
    for token in doc:
        pos_tags.append([token.text, token.tag_])
    
    return pos_tags
```

### Step 4: Dependency Parsing
**Challenge**: Generate dependency parse tree

**Solution**: Use spaCy dependency parser

```python
def get_dependency_tree(tokens):
    """
    Generate dependency parse tree
    
    Args:
        tokens: ["their", "sake", "list", "was", "extensive"]
    
    Returns:
        [["ROOT", 0, 5], ["poss", 3, 1], ["compound", 3, 2], ...]
    """
    text = " ".join(tokens)
    doc = nlp(text)
    
    dependencies = []
    for token in doc:
        if token.dep_ == "ROOT":
            dependencies.append(["ROOT", 0, token.i + 1])
        else:
            dependencies.append([token.dep_, token.head.i + 1, token.i + 1])
    
    return dependencies
```

### Step 5: Create Entity and Sentiment Structures
**Challenge**: Map DimABSA triplets to DESS format

**Solution**: Direct conversion with VA scores

```python
def convert_triplets_to_dess(quadruplets, tokens):
    """
    Convert DimABSA quadruplets to DESS entities and sentiments
    
    Args:
        quadruplets: [{"Aspect": "sake list", "Opinion": "extensive", "VA": "7.83#8.00"}]
        tokens: ["their", "sake", "list", "was", "extensive"]
    
    Returns:
        entities: [{"type": "target", "start": 1, "end": 3}, ...]
        sentiments: [{"type": "7.83#8.00", "head": 0, "tail": 1}]
    """
    entities = []
    sentiments = []
    
    for quad in quadruplets:
        aspect_text = quad["Aspect"]
        opinion_text = quad["Opinion"]
        va_score = quad["VA"]
        
        # Skip NULL aspects/opinions (implicit sentiment)
        if aspect_text == "NULL" or opinion_text == "NULL":
            continue
        
        # Find aspect span
        aspect_span = find_span_indices(tokens, aspect_text)
        if aspect_span is None:
            continue
        
        # Find opinion span
        opinion_span = find_span_indices(tokens, opinion_text)
        if opinion_span is None:
            continue
        
        # Add entities
        aspect_idx = len(entities)
        entities.append({
            "type": "target",
            "start": aspect_span[0],
            "end": aspect_span[1]
        })
        
        opinion_idx = len(entities)
        entities.append({
            "type": "opinion",
            "start": opinion_span[0],
            "end": opinion_span[1]
        })
        
        # Add sentiment with VA score
        sentiments.append({
            "type": va_score,  # Store VA as string "7.83#8.00"
            "head": aspect_idx,
            "tail": opinion_idx
        })
    
    return entities, sentiments
```

---

## Complete Conversion Script

```python
import json
import spacy
from transformers import AutoTokenizer

# Load tools
nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")

def convert_dimabsa_to_dess(dimabsa_jsonl_path, output_json_path):
    """
    Convert DimABSA JSONL file to DESS JSON format
    
    Args:
        dimabsa_jsonl_path: Path to DimABSA .jsonl file
        output_json_path: Path to output DESS .json file
    """
    dess_data = []
    
    with open(dimabsa_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            dimabsa_item = json.loads(line)
            
            # Extract fields
            text = dimabsa_item["Text"]
            orig_id = dimabsa_item["ID"]
            
            # For Subtask 2, we have "Triplet" field
            # For Subtask 3, we have "Quadruplet" field
            triplets = dimabsa_item.get("Triplet", dimabsa_item.get("Quadruplet", []))
            
            # Tokenize (DimABSA is pre-tokenized with spaces)
            tokens = text.split()
            
            # Generate POS tags
            pos_tags = get_pos_tags(tokens)
            
            # Generate dependency tree
            dependencies = get_dependency_tree(tokens)
            
            # Convert triplets to entities and sentiments
            entities, sentiments = convert_triplets_to_dess(triplets, tokens)
            
            # Skip if no valid triplets found
            if not entities or not sentiments:
                continue
            
            # Create DESS format
            dess_item = {
                "tokens": tokens,
                "entities": entities,
                "sentiments": sentiments,
                "pos": pos_tags,
                "dependency": dependencies,
                "orig_id": orig_id
            }
            
            dess_data.append(dess_item)
    
    # Write to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dess_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(dess_data)} samples to DESS format")
    print(f"Saved to: {output_json_path}")

# Usage
convert_dimabsa_to_dess(
    "DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl",
    "DESS/Codebase/data/dimabsa_eng_restaurant/train_dep_triple_polarity_result.json"
)
```

---

## Handling Edge Cases

### Case 1: NULL Aspects/Opinions
**Problem**: DimABSA has implicit sentiments with NULL aspect/opinion
```json
{"Aspect": "NULL", "Opinion": "NULL", "Category": "RESTAURANT#GENERAL", "VA": "6.75#6.38"}
```

**Solution**: Skip these during conversion (DESS requires explicit spans)
- Alternative: Use sentence-level representation (future work)

### Case 2: Multi-word Spans
**Problem**: "sake list" needs to map to tokens[1:3]

**Solution**: Already handled by `find_span_indices()` with multi-token matching

### Case 3: Tokenization Mismatches
**Problem**: DimABSA: "n't" vs DeBERTa: "n", "'", "t"

**Solution**: Use fuzzy matching
```python
def fuzzy_find_span(tokens, span_tokens):
    """Handle tokenization differences"""
    # Try joining with different separators
    span_text = "".join(span_tokens)
    for i in range(len(tokens)):
        for j in range(i+1, len(tokens)+1):
            candidate = "".join(tokens[i:j])
            if candidate == span_text:
                return (i, j)
    return None
```

### Case 4: Multiple Triplets per Sentence
**Problem**: One sentence can have multiple aspect-opinion pairs

**Solution**: Already handled - entities and sentiments are lists

---

## Data Split Strategy

### DimABSA Provides:
1. `train_alltasks.jsonl` - Training data (all subtasks)
2. `dev_task2.jsonl` - Development data (Subtask 2 only)
3. Test data (released during evaluation)

### DESS Requires:
1. `train_dep_triple_polarity_result.json` - Training data
2. `test_dep_triple_polarity_result.json` - Test data (use dev as test)

### Conversion Plan:
```bash
# Convert training data
convert_dimabsa_to_dess(
    "eng_restaurant_train_alltasks.jsonl",
    "data/dimabsa_eng_restaurant/train_dep_triple_polarity_result.json"
)

# Convert dev data (use as test for validation)
convert_dimabsa_to_dess(
    "eng_restaurant_dev_task2.jsonl",
    "data/dimabsa_eng_restaurant/test_dep_triple_polarity_result.json"
)
```

---

## Validation Strategy

### After Conversion, Verify:
1. **Token count matches**: `len(tokens) == len(pos_tags)`
2. **Span validity**: `0 <= start < end <= len(tokens)`
3. **Entity-sentiment alignment**: `head < len(entities)` and `tail < len(entities)`
4. **VA format**: Matches regex `^\d+\.\d{2}#\d+\.\d{2}$`

### Sample Validation Code:
```python
def validate_dess_format(dess_item):
    tokens = dess_item["tokens"]
    entities = dess_item["entities"]
    sentiments = dess_item["sentiments"]
    pos = dess_item["pos"]
    
    # Check token count
    assert len(tokens) == len(pos), "Token count mismatch"
    
    # Check entity spans
    for entity in entities:
        assert 0 <= entity["start"] < entity["end"] <= len(tokens)
    
    # Check sentiment references
    for sentiment in sentiments:
        assert 0 <= sentiment["head"] < len(entities)
        assert 0 <= sentiment["tail"] < len(entities)
        
        # Check VA format
        va = sentiment["type"]
        assert re.match(r'^\d+\.\d{2}#\d+\.\d{2}$', va), f"Invalid VA: {va}"
    
    return True
```

---

## Summary

### Conversion Process:
1. ✅ **Tokenize** DimABSA text (simple split)
2. ✅ **Find spans** using string matching
3. ✅ **Generate POS tags** using spaCy
4. ✅ **Generate dependencies** using spaCy
5. ✅ **Map triplets** to DESS entities/sentiments
6. ✅ **Store VA scores** in sentiment "type" field

### Key Insight:
**We DON'T need DESS's original training data!**
- Convert DimABSA data → DESS format
- Train DESS model on converted data
- Model learns VA regression instead of classification

### Next Steps:
1. Implement conversion script
2. Convert all DimABSA datasets (eng, zho, jpn, etc.)
3. Validate converted data
4. Update DESS data paths in `Parameter.py`
5. Start training!
