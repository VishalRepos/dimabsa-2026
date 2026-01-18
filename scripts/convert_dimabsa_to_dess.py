#!/usr/bin/env python3
"""Convert DimABSA JSONL format to DESS JSON format"""

import json
import argparse
import spacy
from typing import List, Dict, Tuple, Optional

def find_token_span(tokens: List[str], text_span: str, start_from: int = 0) -> Optional[Tuple[int, int]]:
    """Find token indices for a text span"""
    text_tokens = text_span.lower().split()
    tokens_lower = [t.lower() for t in tokens]
    
    for i in range(start_from, len(tokens) - len(text_tokens) + 1):
        if tokens_lower[i:i+len(text_tokens)] == text_tokens:
            return (i, i + len(text_tokens))
    return None

def convert_sample(item: Dict, nlp, use_triplet: bool = False, is_test: bool = False) -> Dict:
    """Convert single DimABSA sample to DESS format"""
    text = item['Text']
    doc = nlp(text)
    
    # Extract tokens
    tokens = [token.text for token in doc]
    
    # Extract POS tags
    pos = [[token.text, token.tag_] for token in doc]
    
    # Extract dependency parsing
    dependency = []
    for token in doc:
        if token.dep_ == "ROOT":
            dependency.append(["ROOT", 0, token.i + 1])
        else:
            dependency.append([token.dep_, token.head.i + 1, token.i + 1])
    
    # Extract entities and sentiments
    entities = []
    sentiments = []
    
    # For test data, return empty entities/sentiments
    if not is_test:
        # Use Triplet or Quadruplet based on flag
        items_key = 'Triplet' if use_triplet else 'Quadruplet'
        items = item.get(items_key, [])
        
        entity_idx = 0
        for triplet in items:
            aspect = triplet.get('Aspect')
            opinion = triplet.get('Opinion')
            va = triplet.get('VA')
            
            # Skip NULL aspects/opinions
            if aspect == 'NULL' or opinion == 'NULL':
                continue
            
            # Find aspect span
            aspect_span = find_token_span(tokens, aspect)
            if aspect_span is None:
                continue
            
            # Find opinion span (search after aspect to avoid duplicates)
            opinion_span = find_token_span(tokens, opinion, aspect_span[1])
            if opinion_span is None:
                # Try searching from beginning if not found after aspect
                opinion_span = find_token_span(tokens, opinion)
            if opinion_span is None:
                continue
            
            # Add entities
            aspect_idx = entity_idx
            entities.append({
                "type": "target",
                "start": aspect_span[0],
                "end": aspect_span[1]
            })
            entity_idx += 1
            
            opinion_idx = entity_idx
            entities.append({
                "type": "opinion",
                "start": opinion_span[0],
                "end": opinion_span[1]
            })
            entity_idx += 1
            
            # Add sentiment with VA score
            sentiments.append({
                "type": va,  # Store VA as string "V.VV#A.AA"
                "head": aspect_idx,
                "tail": opinion_idx
            })
    
    return {
        "tokens": tokens,
        "entities": entities,
        "sentiments": sentiments,
        "pos": pos,
        "dependency": dependency,
        "orig_id": item['ID']
    }

def convert_dataset(input_path: str, output_path: str, use_triplet: bool = False, is_test: bool = False):
    """Convert entire DimABSA dataset to DESS format"""
    print(f"Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    
    print(f"Reading {input_path}...")
    samples = []
    skipped = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            item = json.loads(line)
            try:
                converted = convert_sample(item, nlp, use_triplet, is_test)
                # For test data, always add; for training, only if entities exist
                if is_test or converted['entities']:
                    samples.append(converted)
                else:
                    skipped += 1
            except Exception as e:
                print(f"Warning: Skipped line {line_num} ({item.get('ID', 'unknown')}): {e}")
                skipped += 1
    
    print(f"Converted {len(samples)} samples ({skipped} skipped)")
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Done! Output: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert DimABSA JSONL to DESS JSON format')
    parser.add_argument('--input', required=True, help='Input DimABSA JSONL file')
    parser.add_argument('--output', required=True, help='Output DESS JSON file')
    parser.add_argument('--triplet', action='store_true', help='Use Triplet field instead of Quadruplet')
    parser.add_argument('--test', action='store_true', help='Test data (no labels)')
    
    args = parser.parse_args()
    
    convert_dataset(args.input, args.output, args.triplet, args.test)

if __name__ == '__main__':
    main()
