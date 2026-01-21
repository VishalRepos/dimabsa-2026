#!/bin/bash
# Training script for Laptop domain

cd /Users/vishal.thenuwara/Documents/MSC/Research/Coding/Competition/DimABSANew
source venv/bin/activate
cd Pipeline-DeBERTa

# Laptop domain training
echo "=== Training on Laptop Domain ==="
python run_task2\&3_trainer_multilingual.py \
  --task 2 \
  --domain lap \
  --language eng \
  --train_data track_a/subtask_2/eng/eng_laptop_train_alltasks.jsonl \
  --infer_data track_a/subtask_2/eng/eng_laptop_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode train \
  --epoch_num 3 \
  --batch_size 8 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --inference_beta 0.9

echo ""
echo "=== Training Complete ==="
echo "Model saved to: ./model/task2_lap_eng.pth"
echo "Predictions saved to: ./tasks/subtask_2/pred_eng_laptop.jsonl"
