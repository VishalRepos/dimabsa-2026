# DimABSA 2026 - DESS Model with VA Regression

Dimensional Aspect-Based Sentiment Analysis using DESS (Dual-channel Enhanced Sentiment Span) model adapted for Valence-Arousal regression.

## 🎯 Task

**Track A - Subtask 2**: Dimensional Aspect Sentiment Triplet Extraction (DimASTE)

Extract (Aspect, Opinion, VA) triplets from text, where VA represents continuous Valence-Arousal scores.

## 📊 Dataset

- **Training**: 3,727 samples (Restaurant + Laptop domains)
- **Test**: 400 samples
- **Source**: DimABSA 2026 Competition - Subtask 1 data
- **Format**: DESS JSON with tokens, entities, sentiments, POS tags, dependencies

## 🏗️ Model Architecture

**Base Model**: DESS (Dual-channel Enhanced Sentiment Span)

**Modifications**:
- Sentiment classifier → VA regression head (2 outputs)
- Loss function: BCEWithLogitsLoss → MSELoss
- Output: Continuous VA scores [1.0, 9.0]

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR-USERNAME/dimabsa-2026.git
cd dimabsa-2026
```

### 2. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Prepare Data
```bash
# Data already converted and included in repository
# Located in: DESS/Codebase/data/dimabsa_combined/
```

### 4. Train on Kaggle
See [KAGGLE_SETUP_GUIDE.md](KAGGLE_SETUP_GUIDE.md) for detailed instructions.

**Quick steps**:
1. Upload repository to Kaggle as dataset
2. Use `kaggle_training.ipynb` notebook
3. Enable GPU and run training
4. Download trained model

## 📁 Repository Structure

```
dimabsa-2026/
├── DESS/
│   └── Codebase/
│       ├── models/              # Model architecture
│       ├── trainer/             # Training utilities
│       ├── data/                # Converted datasets
│       └── Parameter.py         # Configuration
├── scripts/
│   ├── convert_dimabsa_to_dess.py    # Data converter
│   └── prepare_kaggle_upload.sh      # Kaggle package creator
├── Testing/                     # Test scripts and reports
├── kaggle_training.ipynb        # Kaggle training notebook
├── requirements.txt             # Python dependencies
├── KAGGLE_SETUP_GUIDE.md       # Kaggle setup instructions
└── README.md                    # This file
```

## 🔧 Key Features

- ✅ VA regression instead of sentiment classification
- ✅ Combined restaurant + laptop domains
- ✅ MSE loss for continuous prediction
- ✅ Kaggle-ready training notebook
- ✅ Comprehensive testing suite
- ✅ Data conversion pipeline

## 📈 Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | DeBERTa-v3-base |
| Batch Size | 4 |
| Epochs | 10 |
| Learning Rate | 5e-5 |
| Training Samples | 3,727 |
| GPU | T4 x2 / P100 |
| Training Time | ~2-3 hours |

## 🧪 Testing

All phases tested and validated:

```bash
# Phase 1: Data Conversion
python Testing/Phase1/test_conversion.py

# Phase 2: Model Modifications
python Testing/Phase2/test_model.py

# Combined Testing
python Testing/test_phase1_phase2_combined.py
```

**Results**: 8/8 tests passed (100%)

## 📊 Data Statistics

**Combined Dataset**:
- Training: 3,727 samples
- Test: 400 samples
- Avg entities/sample: 3.06
- Avg sentiments/sample: 1.53
- Total VA pairs: 5,694
- Avg Valence: 6.40
- Avg Arousal: 7.13

## 🎓 Model Details

### Input Format (DESS)
```json
{
  "tokens": ["the", "food", "was", "great"],
  "entities": [
    {"type": "target", "start": 1, "end": 2},
    {"type": "opinion", "start": 3, "end": 4}
  ],
  "sentiments": [
    {"type": "7.50#7.62", "head": 0, "tail": 1}
  ],
  "pos": [...],
  "dependency": [...]
}
```

### Output Format (DimABSA)
```json
{
  "ID": "...",
  "Triplet": [
    {
      "Aspect": "food",
      "Opinion": "great",
      "VA": "7.50#7.62"
    }
  ]
}
```

## 📝 Documentation

- [KAGGLE_SETUP_GUIDE.md](KAGGLE_SETUP_GUIDE.md) - Detailed Kaggle setup
- [QUICK_START_KAGGLE.md](QUICK_START_KAGGLE.md) - Quick reference
- [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md) - Data conversion details
- [PHASE2_COMPLETE.md](PHASE2_COMPLETE.md) - Model modifications
- [PHASE3_KAGGLE_READY.md](PHASE3_KAGGLE_READY.md) - Training setup
- [SUBTASKS_ANALYSIS.md](SUBTASKS_ANALYSIS.md) - Task analysis
- [DATA_CORRECTION.md](DATA_CORRECTION.md) - Data source verification

## 🏆 Competition

**DimABSA 2026 Shared Task**
- Track A: Dimensional ABSA
- Subtask 2: Triplet Extraction
- Metric: Continuous F1
- Website: [CodaBench](https://www.codabench.org/competitions/10918/)

## 📄 License

This project uses the DESS model architecture. Original DESS paper and code should be cited appropriately.

## 🙏 Acknowledgments

- DimABSA 2026 organizers
- DESS model authors
- Original dataset creators

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Status**: ✅ Ready for Kaggle Training

*Last updated: 2026-01-18*
