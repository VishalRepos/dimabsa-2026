#!/bin/bash
# Quick training test - 1 epoch on restaurant data

cd /Users/vishal.thenuwara/Documents/MSC/Research/Coding/Competition/DimABSANew
source venv/bin/activate
cd Pipeline-DeBERTa

echo "=== Quick Training Test (1 epoch, Restaurant) ==="
echo "This will take ~10-15 minutes..."
echo ""

python run_task2\&3_trainer_multilingual.py \
  --task 2 \
  --domain res \
  --language eng \
  --train_data track_a/subtask_2/eng/eng_restaurant_train_alltasks.jsonl \
  --infer_data track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl \
  --bert_model_type microsoft/deberta-v3-base \
  --mode train \
  --epoch_num 1 \
  --batch_size 4 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --inference_beta 0.9 \
  --gpu False

echo ""
echo "=== Test Complete ==="
echo "Check:"
echo "  - Log: ./log/AOC.log"
echo "  - Model: ./model/task2_res_eng.pth"
echo "  - Predictions: ./tasks/subtask_2/pred_eng_restaurant.jsonl"
