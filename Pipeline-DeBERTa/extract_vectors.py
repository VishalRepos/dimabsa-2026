"""
Vector Extraction Utility for Pipeline-DeBERTa
Extracts [CLS] token embeddings from trained model
"""

import torch
import json
import os
from tqdm import tqdm
import numpy as np


def extract_vectors_from_data(model, data_loader, device, output_file, save_every=100):
    """
    Extract [CLS] vectors for all samples in data_loader
    
    Args:
        model: Trained DimABSA model
        data_loader: DataLoader with samples
        device: torch device
        output_file: Path to save JSONL file
        save_every: Save to disk every N samples (to avoid memory issues)
    """
    model.eval()
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Open file in append mode
    with open(output_file, 'w') as f:
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting vectors"):
                # Get batch data
                query_tensor = batch['query_tensor'].to(device)
                query_mask = batch['query_mask'].to(device)
                query_seg = batch['query_seg'].to(device)
                
                # Extract [CLS] vectors
                cls_vectors = model(query_tensor, query_mask, query_seg, 
                                   step='A', return_vectors=True)
                
                # Convert to numpy and save
                cls_vectors = cls_vectors.cpu().numpy()
                
                # Save each sample
                for i in range(cls_vectors.shape[0]):
                    vector_data = {
                        'id': batch['id'][i] if 'id' in batch else f"sample_{sample_count}",
                        'text': batch['text'][i] if 'text' in batch else "",
                        'vector': cls_vectors[i].tolist(),
                    }
                    
                    # Add labels if available (training data)
                    if 'aspect' in batch:
                        vector_data['aspect'] = batch['aspect'][i]
                    if 'opinion' in batch:
                        vector_data['opinion'] = batch['opinion'][i]
                    if 'valence' in batch:
                        vector_data['valence'] = float(batch['valence'][i])
                    if 'arousal' in batch:
                        vector_data['arousal'] = float(batch['arousal'][i])
                    
                    # Write to file
                    f.write(json.dumps(vector_data) + '\n')
                    sample_count += 1
    
    print(f"✓ Saved {sample_count} vectors to {output_file}")
    return sample_count


def extract_vectors_simple(model, tokenizer, texts, ids, device, output_file):
    """
    Simple vector extraction from list of texts
    
    Args:
        model: Trained DimABSA model
        tokenizer: DeBERTa tokenizer
        texts: List of text strings
        ids: List of sample IDs
        device: torch device
        output_file: Path to save JSONL file
    """
    model.eval()
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w') as f:
        with torch.no_grad():
            for idx, (text, sample_id) in enumerate(tqdm(zip(texts, ids), total=len(texts), desc="Extracting")):
                # Tokenize
                encoded = tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                
                query_tensor = encoded['input_ids'].to(device)
                query_mask = encoded['attention_mask'].to(device)
                query_seg = torch.zeros_like(query_tensor).to(device)
                
                # Extract vector
                cls_vector = model(query_tensor, query_mask, query_seg, 
                                  step='A', return_vectors=True)
                
                # Save
                vector_data = {
                    'id': sample_id,
                    'text': text,
                    'vector': cls_vector[0].cpu().numpy().tolist()
                }
                
                f.write(json.dumps(vector_data) + '\n')
    
    print(f"✓ Saved {len(texts)} vectors to {output_file}")


def load_vectors(jsonl_file):
    """
    Load vectors from JSONL file
    
    Returns:
        dict: {'ids': [...], 'texts': [...], 'vectors': np.array, 'metadata': [...]}
    """
    ids = []
    texts = []
    vectors = []
    metadata = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            ids.append(data['id'])
            texts.append(data.get('text', ''))
            vectors.append(data['vector'])
            
            # Store other metadata
            meta = {k: v for k, v in data.items() if k not in ['id', 'text', 'vector']}
            metadata.append(meta)
    
    return {
        'ids': ids,
        'texts': texts,
        'vectors': np.array(vectors),
        'metadata': metadata
    }


if __name__ == "__main__":
    import argparse
    from transformers import DebertaV2Tokenizer
    from DimABSAModel import DimABSA
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--data_file', required=True, help='Input JSONL data file')
    parser.add_argument('--output_file', required=True, help='Output JSONL vector file')
    parser.add_argument('--bert_model', default='microsoft/deberta-v3-base')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_category', type=int, default=30, help='Number of categories (default: 30)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    
    # Load checkpoint first to detect num_category
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # Detect num_category from checkpoint
    if 'net' in checkpoint:
        state_dict = checkpoint['net']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Get num_category from classifier_category.weight shape
    if 'classifier_category.weight' in state_dict:
        detected_num_category = state_dict['classifier_category.weight'].shape[0]
        print(f"Detected num_category from checkpoint: {detected_num_category}")
        num_category = detected_num_category
    else:
        num_category = args.num_category
        print(f"Using default num_category: {num_category}")
    
    # Create model with correct num_category
    model = DimABSA(args.hidden_size, args.bert_model, num_category=num_category)
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    
    # Load tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.bert_model)
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    texts = []
    ids = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            ids.append(data['ID'])
            texts.append(data.get('Sentence', data.get('Text', '')))
    
    # Extract vectors
    print(f"Extracting vectors for {len(texts)} samples...")
    extract_vectors_simple(model, tokenizer, texts, ids, args.device, args.output_file)
    
    print("✓ Done!")
