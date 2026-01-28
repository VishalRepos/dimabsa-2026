# DimABSA 2026 - Documentation Index

**Competition**: DimABSA 2026 Shared Task - Track A, Subtask 2  
**Task**: Extract (Aspect, Opinion, VA) triplets with continuous Valence-Arousal scores  
**Repository**: https://github.com/VishalRepos/dimabsa-2026.git

---

## 📚 Documentation Structure

### Core Documentation
1. **[00-COMPETITION.md](00-COMPETITION.md)** - Competition overview and task description
2. **[01-DATA_FORMAT.md](01-DATA_FORMAT.md)** - Input/output data specifications

### Approach-Specific Documentation

#### DESS Approach (Primary)
3. **[02-DESS-ARCHITECTURE.md](02-DESS-ARCHITECTURE.md)** - DESS model architecture and modifications
4. **[03-DESS-TRAINING.md](03-DESS-TRAINING.md)** - Training setup and execution
5. **[04-DESS-INFERENCE.md](04-DESS-INFERENCE.md)** - Inference and submission

#### Pipeline-DeBERTa Approach (Alternative)
6. **[05-PIPELINE-ARCHITECTURE.md](05-PIPELINE-ARCHITECTURE.md)** - Pipeline model architecture
7. **[06-PIPELINE-TRAINING.md](06-PIPELINE-TRAINING.md)** - Training setup and execution
8. **[07-PIPELINE-INFERENCE.md](07-PIPELINE-INFERENCE.md)** - Inference and submission

### Comparison
9. **[08-APPROACH-COMPARISON.md](08-APPROACH-COMPARISON.md)** - Side-by-side comparison of both approaches

---

## 🚀 Quick Start

### For DESS (Recommended)
```bash
# 1. Training
cd DESS/Codebase
python train.py --dataset dimabsa_combined --epochs 10 --batch_size 4

# 2. Inference
python predict.py --model_path /path/to/model --test_data data/dimabsa_combined/test.json
```

### For Pipeline-DeBERTa
```bash
# 1. Training
cd Pipeline-DeBERTa
python run_task2&3_trainer_multilingual.py --task 2 --domain res --mode train

# 2. Inference
python run_task2&3_trainer_multilingual.py --task 2 --domain res --mode inference
```

---

## 📊 Current Status

| Approach | Status | F1 Score | Notes |
|----------|--------|----------|-------|
| **DESS** | ✅ Trained | ~8.22% | Primary approach, ready for submission |
| **Pipeline-DeBERTa** | 🔄 Setup | TBD | Alternative/backup approach |

---

## 📁 Repository Structure

```
DimABSANew/
├── docs/                          # ← Consolidated documentation (YOU ARE HERE)
├── DESS/Codebase/                 # DESS approach implementation
├── Pipeline-DeBERTa/              # Pipeline approach implementation
├── scripts/                       # Utility scripts
├── DimABSA2026/task-dataset/     # Competition data
└── README.md                      # Main project README
```

---

## 🎯 Which Approach to Use?

**Use DESS if**: You want the best performance, single forward pass, rich features  
**Use Pipeline-DeBERTa if**: You want simpler architecture, easier debugging, MRC-based approach

See [08-APPROACH-COMPARISON.md](08-APPROACH-COMPARISON.md) for detailed comparison.

---

**Last Updated**: 2026-01-25
