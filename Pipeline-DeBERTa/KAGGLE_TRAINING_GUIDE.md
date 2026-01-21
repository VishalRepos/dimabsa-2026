# Kaggle Training Guide for Pipeline-DeBERTa

## 📋 Prerequisites

1. Kaggle account
2. GPU quota available (T4 or P100)
3. Pipeline-DeBERTa code ready

## 🚀 Quick Start

### Option 1: Upload Code as Dataset (Recommended)

1. **Prepare Upload**:
   ```bash
   cd /Users/vishal.thenuwara/Documents/MSC/Research/Coding/Competition/DimABSANew
   zip -r pipeline-deberta-code.zip Pipeline-DeBERTa/ \
       -x "Pipeline-DeBERTa/data/*" \
       -x "Pipeline-DeBERTa/model/*" \
       -x "Pipeline-DeBERTa/log/*" \
       -x "Pipeline-DeBERTa/tasks/*" \
       -x "Pipeline-DeBERTa/__pycache__/*" \
       -x "Pipeline-DeBERTa/*.log"
   ```

2. **Upload to Kaggle**:
   - Go to https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload `pipeline-deberta-code.zip`
   - Title: "Pipeline-DeBERTa Code"
   - Make it private

3. **Create Notebook**:
   - Go to https://www.kaggle.com/code
   - Click "New Notebook"
   - Settings → Accelerator → GPU T4
   - Add Data → Your Datasets → "Pipeline-DeBERTa Code"
   - Upload `kaggle_training.ipynb` or copy cells

4. **Run Training**:
   - Execute all cells
   - Monitor progress
   - Download results when complete

### Option 2: Direct Upload to Notebook

1. **Create Notebook**:
   - New Notebook on Kaggle
   - Enable GPU T4

2. **Upload Files**:
   - Click "Add Data" → "Upload"
   - Upload these files:
     - `DimABSAModel.py`
     - `Utils.py`
     - `DataProcess.py`
     - `run_task2&3_trainer_multilingual.py`

3. **Copy Notebook Cells**:
   - Open `kaggle_training.ipynb`
   - Copy all cells to Kaggle notebook

4. **Run**

## 📊 Expected Timeline

| Step | Time | Description |
|------|------|-------------|
| Setup | 5 min | Install dependencies, download data |
| Restaurant Training | 30-45 min | 2,284 samples, 3 epochs |
| Laptop Training | 60-90 min | 4,076 samples, 3 epochs |
| **Total** | **~2-3 hours** | Full pipeline |

## 💾 Output Files

After training completes:

```
results/
├── task2_res_eng.pth          # Restaurant model
├── task2_lap_eng.pth          # Laptop model
├── pred_eng_restaurant.jsonl  # Restaurant predictions
├── pred_eng_laptop.jsonl      # Laptop predictions
└── AOC.log                    # Training logs
```

## 🔍 Monitoring Training

Watch for these in the logs:

```
Epoch 1/3
  Triplet - Precision: X.XX  Recall: X.XX  F1: X.XX
  Aspect - Precision: X.XX  Recall: X.XX  F1: X.XX
  Opinion - Precision: X.XX  Recall: X.XX  F1: X.XX
```

**Good signs**:
- F1 scores increasing across epochs
- Precision and recall balanced
- No NaN or inf values

**Warning signs**:
- F1 stuck at 0
- Loss not decreasing
- Out of memory errors (reduce batch_size to 4)

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size in training command
--batch_size 4  # instead of 8
```

### Slow Training
- Verify GPU is enabled (Settings → Accelerator)
- Check GPU usage: `!nvidia-smi`

### Import Errors
```python
# Re-run dependency installation
!pip install -q transformers torch sentencepiece protobuf
```

### Data Download Fails
```python
# Try alternative download
!git clone https://github.com/DimABSA/DimABSA2026.git
!cp DimABSA2026/task-dataset/track_a/subtask_2/eng/*.jsonl data/track_a/subtask_2/eng/
```

## 📈 Expected Performance

Based on starter kit baseline + DeBERTa upgrade:

| Domain | Expected F1 | Baseline (BERT) |
|--------|-------------|-----------------|
| Restaurant | 15-25% | ~10-15% |
| Laptop | 12-20% | ~8-12% |

**Much better than DESS (8.22%)!**

## 📥 Download Results

After training:

1. Click "Save Version" → "Save & Run All"
2. Wait for completion
3. Go to "Output" tab
4. Download `pipeline_deberta_results.zip`

Or download individual files:
- Right-click file → Download

## 🎯 Next Steps After Training

1. **Validate predictions** (done in notebook)
2. **Check F1 scores** in logs
3. **Download models** for local inference
4. **Submit to competition** (if test data available)

## 💡 Tips

- **Save frequently**: Click "Save Version" to checkpoint
- **Monitor GPU**: Use `!nvidia-smi` to check utilization
- **Adjust beta**: If too many/few predictions, change `--inference_beta` (0.8-0.95)
- **Try more epochs**: If F1 still improving, increase `--epoch_num` to 5

## 📞 Support

If issues arise:
1. Check Kaggle notebook output
2. Review error messages
3. Verify GPU is enabled
4. Check data downloaded correctly

---

**Ready to train!** 🚀

Upload the code and run `kaggle_training.ipynb` on Kaggle with GPU T4.
