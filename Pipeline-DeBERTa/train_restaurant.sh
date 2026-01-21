#!/bin/bash
# Training script for Pipeline-DeBERTa

cd /Users/vishal.thenuwara/Documents/MSC/Research/Coding/Competition/DimABSANew
source venv/bin/activate
cd Pipeline-DeBERTa

# Restaurant domain training
echo "=== Training on Restaurant Domain ==="
python run_task2\&3_trainer_multilingual.py \
  --task 2 \
  --domain res \
  --language eng \
  --train_data track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
  --infer_data track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode train \
  --epoch_num 3 \
  --batch_size 8 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --inference_beta 0.9

echo ""
echo "=== Training Complete ==="
echo "Model saved to: ./model/task2_res_eng.pth"
echo "Predictions saved to: ./tasks/subtask_2/pred_eng_restaurant.jsonl"
