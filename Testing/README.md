# Testing Strategy

This folder contains testing scripts and results for each phase of the DimABSA project.

## Structure

```
Testing/
├── Phase1/                    # Data Conversion Testing
│   ├── test_conversion.py     # Automated test suite
│   ├── test_results.json      # Raw test results
│   ├── TEST_REPORT.md         # Detailed analysis
│   └── SUMMARY.md             # Quick summary
├── Phase2/                    # Model Modification Testing (TBD)
├── Phase3/                    # Training Testing (TBD)
├── Phase4/                    # Inference Testing (TBD)
└── README.md                  # This file
```

## Testing Philosophy

Each phase includes:
1. **Automated tests** - Scripts to validate functionality
2. **Results** - JSON files with metrics
3. **Reports** - Detailed analysis with examples
4. **Summary** - Quick pass/fail assessment

## Phase 1: Data Conversion ✅ COMPLETE

**Status**: PASS (99.88% accuracy)

**Tests**:
- Span reconstruction accuracy
- VA score preservation
- Linguistic features coverage
- Dataset statistics

**Run tests**:
```bash
python Testing/Phase1/test_conversion.py
```

**Results**: See `Phase1/SUMMARY.md`

## Phase 2: Model Modification (Upcoming)

**Planned tests**:
- VA regression head output shape
- Loss function computation
- Forward pass validation
- Gradient flow check

## Phase 3: Training (Upcoming)

**Planned tests**:
- Training loop execution
- Loss convergence
- Checkpoint saving
- Validation metrics

## Phase 4: Inference (Upcoming)

**Planned tests**:
- Prediction format validation
- Reverse conversion accuracy
- Submission file format
- End-to-end pipeline

## Running All Tests

```bash
# Phase 1
python Testing/Phase1/test_conversion.py

# Phase 2 (when ready)
python Testing/Phase2/test_model.py

# Phase 3 (when ready)
python Testing/Phase3/test_training.py

# Phase 4 (when ready)
python Testing/Phase4/test_inference.py
```

## Success Criteria

Each phase must achieve:
- ✅ All automated tests pass
- ✅ Accuracy/quality metrics > 95%
- ✅ No critical errors
- ✅ Documentation complete

---

*Last updated: 2026-01-18*
