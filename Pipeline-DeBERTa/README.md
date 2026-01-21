# Pipeline-DeBERTa for DimABSA 2026

## Overview
Pipeline-based approach using DeBERTa-v3-base for Subtask 2 (Triplet Extraction).

Adapted from official starter kit with BERT replaced by DeBERTa-v3-base.

## Structure
```
Pipeline-DeBERTa/
├── model/              # Saved model checkpoints
├── log/                # Training logs
├── tasks/
│   ├── subtask_2/     # Task 2 predictions
│   └── subtask_3/     # Task 3 predictions
├── data/              # Symlink to dataset
├── DimABSAModel.py    # Model architecture (DeBERTa)
├── Utils.py           # Utilities
├── DataProcess.py     # Data processing
└── run_task2&3_trainer_multilingual.py  # Main script
```

## Quick Start

### Training
```bash
python run_task2&3_trainer_multilingual.py \
  --task 2 \
  --domain res \
  --language eng \
  --train_data data/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
  --infer_data data/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode train \
  --epoch_num 3 \
  --batch_size 8
```

### Inference Only
```bash
python run_task2&3_trainer_multilingual.py \
  --task 2 \
  --domain res \
  --language eng \
  --train_data data/track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
  --infer_data data/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode inference
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| --task | 2 | Task 2 (triplets) or 3 (quadruplets) |
| --domain | res | Domain: res, lap, hot, fin |
| --language | eng | Language: eng, zho, jpn |
| --bert_model_type | microsoft/deberta-v3-base | Model name |
| --epoch_num | 3 | Training epochs |
| --batch_size | 8 | Batch size |
| --learning_rate | 1e-3 | LR for non-BERT params |
| --tuning_bert_rate | 1e-5 | LR for DeBERTa |
| --inference_beta | 0.9 | Confidence threshold |

## Output Format

Predictions saved to `tasks/subtask_2/pred_{language}_{domain}.jsonl`

Format:
```json
{
  "ID": "...",
  "Triplet": [
    {"Aspect": "...", "Opinion": "...", "VA": "V.VV#A.AA"}
  ]
}
```

## Status
- [x] Project structure created
- [ ] Model adapted for DeBERTa
- [ ] Training tested
- [ ] Inference validated
