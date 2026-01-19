"""
Inference script for DimABSA VA regression model
Generates predictions on test data and creates submission file
"""

import argparse
import json
import os
import torch
from transformers import AutoTokenizer
from models.D2E2S_Model import D2E2SModel
from trainer.input_reader import JsonInputReader
from trainer.entities import Dataset
from torch.utils.data import DataLoader
from trainer import sampling, util
from tqdm import tqdm


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load config
    from transformers import AutoConfig
    import argparse
    config = AutoConfig.from_pretrained(model_path)
    
    # Create args object with required parameters
    args = argparse.Namespace(
        size_embedding=25,
        prop_drop=0.1,
        freeze_transformer=False,
        lstm_layers=1,
        lstm_dim=384,
        hidden_dim=384,
        emb_dim=768,
        mem_dim=768,
        gcn_dim=300,
        gcn_dropout=0.2,
        deberta_feature_dim=768,
        drop_out_rate=0.5,
        is_bidirect=True,
        batch_size=4,
        pretrained_deberta_name="microsoft/deberta-v3-base",
        attention_heads=1,
        num_layers=2,
        use_gated=False,
        span_generator="Max",
        pooling="avg"
    )
    
    # Load model
    model = D2E2SModel(
        config=config,
        cls_token=tokenizer.convert_tokens_to_ids("[CLS]"),
        sentiment_types=1,
        entity_types=3,
        args=args
    )
    
    # Load weights
    import torch
    import os
    
    # Check for model file
    if os.path.exists(f"{model_path}/pytorch_model.bin"):
        state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location=device)
    elif os.path.exists(f"{model_path}/model.safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(f"{model_path}/model.safetensors")
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}")
    
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    return model, tokenizer


def predict(model, dataset, tokenizer, device, batch_size=4):
    """Run inference on dataset"""
    dataset.switch_mode(Dataset.EVAL_MODE)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=sampling.collate_fn_padding,
    )
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating predictions"):
            batch = util.to_device(batch, device)
            
            # Forward pass
            entity_clf, senti_clf, rels = model(
                encodings=batch["encodings"],
                context_masks=batch["context_masks"],
                entity_masks=batch["entity_masks"],
                entity_sizes=batch["entity_sizes"],
                entity_spans=batch["entity_spans"],
                entity_sample_masks=batch["entity_sample_masks"],
                adj=batch["adj"],
                evaluate=True,
            )
            
            # Process predictions for each sample in batch
            batch_size = entity_clf.shape[0]
            for i in range(batch_size):
                sample_pred = process_sample_prediction(
                    entity_clf[i],
                    senti_clf[i],
                    rels[i],
                    batch["entity_spans"][i],
                    batch["tokens"][i],
                    tokenizer
                )
                predictions.append(sample_pred)
    
    return predictions


def process_sample_prediction(entity_clf, senti_clf, rels, entity_spans, tokens, tokenizer):
    """Process predictions for a single sample"""
    # Get predicted entity types
    entity_types = entity_clf.argmax(dim=-1)
    
    # Filter valid entities (non-zero type)
    valid_entity_mask = entity_types > 0
    valid_entity_indices = valid_entity_mask.nonzero().view(-1)
    
    if len(valid_entity_indices) == 0:
        return []
    
    # Get sentiment predictions
    senti_va_scores = senti_clf  # [pairs, 2] - VA scores
    senti_magnitudes = torch.norm(senti_va_scores, dim=-1)
    
    # Filter valid sentiments (magnitude > threshold)
    threshold = 0.1
    valid_senti_mask = senti_magnitudes > threshold
    valid_senti_indices = valid_senti_mask.nonzero().view(-1)
    
    triplets = []
    for idx in valid_senti_indices:
        rel = rels[idx]  # [head_idx, tail_idx]
        head_idx, tail_idx = rel[0].item(), rel[1].item()
        
        # Check if both entities are valid
        if head_idx >= len(entity_spans) or tail_idx >= len(entity_spans):
            continue
        
        head_span = entity_spans[head_idx]
        tail_span = entity_spans[tail_idx]
        
        # Get entity text
        aspect_text = " ".join(tokens[head_span[0]:head_span[1]])
        opinion_text = " ".join(tokens[tail_span[0]:tail_span[1]])
        
        # Get VA scores
        va_scores = senti_va_scores[idx]
        valence = va_scores[0].item()
        arousal = va_scores[1].item()
        va_string = f"{valence:.2f}#{arousal:.2f}"
        
        triplets.append({
            "Aspect": aspect_text,
            "Opinion": opinion_text,
            "VA": va_string
        })
    
    return triplets


def create_submission(predictions, test_data, output_path):
    """Create submission file in DimABSA format"""
    submission = []
    
    for i, (pred_triplets, test_sample) in enumerate(zip(predictions, test_data)):
        sample_id = test_sample.get("ID", f"sample_{i}")
        
        submission.append({
            "ID": sample_id,
            "Triplet": pred_triplets
        })
    
    # Save submission
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Submission saved to: {output_path}")
    print(f"  Total samples: {len(submission)}")
    print(f"  Total triplets: {sum(len(s['Triplet']) for s in submission)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data JSON")
    parser.add_argument("--types_path", type=str, default="data/types_va.json", help="Path to types JSON")
    parser.add_argument("--output", type=str, default="submission.json", help="Output submission file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print("="*60)
    print("DimABSA Inference - VA Regression")
    print("="*60)
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.device)
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    input_reader = JsonInputReader(
        args.types_path,
        tokenizer,
        max_span_size=10
    )
    input_reader.read({
        "test": args.test_data
    })
    test_dataset = input_reader.get_dataset("test")
    
    with open(args.test_data, 'r') as f:
        test_data_raw = json.load(f)
    
    print(f"✓ Loaded {len(test_dataset)} test samples")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predict(model, test_dataset, tokenizer, args.device, args.batch_size)
    
    # Create submission file
    print("\nCreating submission file...")
    create_submission(predictions, test_data_raw, args.output)
    
    print("\n" + "="*60)
    print("✓ Inference completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
