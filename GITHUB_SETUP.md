# GitHub Repository Setup Guide

## Step 1: Initialize Git Repository

```bash
cd /Users/vishal.thenuwara/Documents/MSC/Research/Coding/Competition/DimABSANew

# Initialize git (if not already done)
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: DimABSA 2026 - DESS with VA Regression

- Phase 1: Data conversion (subtask_1, restaurant + laptop)
- Phase 2: Model modifications (VA regression)
- Phase 3: Kaggle training setup
- All tests passing (8/8)
- 3,727 training samples ready"
```

## Step 2: Create GitHub Repository

### Option A: Using GitHub Web Interface

1. Go to https://github.com/new
2. Repository name: `dimabsa-2026`
3. Description: "DimABSA 2026 - DESS Model with VA Regression for Aspect Sentiment Triplet Extraction"
4. Choose: Public or Private
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

### Option B: Using GitHub CLI

```bash
# Install GitHub CLI if needed
# brew install gh  # macOS
# Or download from: https://cli.github.com/

# Login
gh auth login

# Create repository
gh repo create dimabsa-2026 --public --source=. --remote=origin --push
```

## Step 3: Connect Local Repository to GitHub

```bash
# Add remote (replace YOUR-USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR-USERNAME/dimabsa-2026.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Verify Upload

Check that these files are on GitHub:
- ✅ README.md
- ✅ .gitignore
- ✅ requirements.txt
- ✅ kaggle_training.ipynb
- ✅ DESS/Codebase/ (all model and trainer files)
- ✅ scripts/ (conversion and preparation scripts)
- ✅ Testing/ (test scripts and reports)
- ✅ Documentation files (*.md)

## Step 5: Check Data Files

### Data File Sizes
```bash
# Check size of data files
du -h DESS/Codebase/data/dimabsa_combined/*.json
```

**If files are too large for GitHub (>100 MB)**:

### Option A: Use Git LFS
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.json"
git lfs track "DESS/Codebase/data/**/*.json"

# Add .gitattributes
git add .gitattributes

# Commit and push
git commit -m "Add Git LFS for large data files"
git push
```

### Option B: Exclude from Git
```bash
# Add to .gitignore
echo "DESS/Codebase/data/dimabsa_combined/*.json" >> .gitignore

# Commit
git add .gitignore
git commit -m "Exclude large data files from git"
git push
```

**Note**: If excluding data, document how to obtain it in README.

## Step 6: Add Repository Badges (Optional)

Add to top of README.md:

```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
```

## Step 7: Create Release (Optional)

After successful training:

```bash
# Tag the release
git tag -a v1.0 -m "Initial release: Trained model for DimABSA Subtask 2"

# Push tag
git push origin v1.0
```

Then on GitHub:
1. Go to "Releases"
2. Click "Create a new release"
3. Select tag v1.0
4. Add release notes
5. Attach trained model (if < 2GB)

## Step 8: Use in Kaggle

### Method 1: Direct GitHub Integration

1. In Kaggle notebook, click "Add Data"
2. Select "GitHub"
3. Enter: `YOUR-USERNAME/dimabsa-2026`
4. Kaggle will clone the repository

### Method 2: Create Kaggle Dataset from GitHub

1. Clone repository locally
2. Create ZIP: `bash scripts/prepare_kaggle_upload.sh`
3. Upload ZIP to Kaggle as dataset

## Repository Structure on GitHub

```
dimabsa-2026/
├── .gitignore
├── README.md
├── requirements.txt
├── kaggle_training.ipynb
├── DESS/
│   └── Codebase/
│       ├── models/
│       ├── trainer/
│       ├── data/
│       └── Parameter.py
├── scripts/
│   ├── convert_dimabsa_to_dess.py
│   └── prepare_kaggle_upload.sh
├── Testing/
│   ├── Phase1/
│   ├── Phase2/
│   └── test_phase1_phase2_combined.py
└── docs/
    ├── KAGGLE_SETUP_GUIDE.md
    ├── QUICK_START_KAGGLE.md
    ├── PHASE1_COMPLETE.md
    ├── PHASE2_COMPLETE.md
    ├── PHASE3_KAGGLE_READY.md
    ├── SUBTASKS_ANALYSIS.md
    └── DATA_CORRECTION.md
```

## Troubleshooting

### Large Files Error
```
remote: error: File DESS/Codebase/data/... is 123.45 MB; this exceeds GitHub's file size limit of 100.00 MB
```

**Solution**: Use Git LFS or exclude from git (see Step 5)

### Authentication Failed
```bash
# Use personal access token
# Generate at: https://github.com/settings/tokens
# Use token as password when pushing
```

### Already Exists Error
```bash
# If repository already exists, use:
git remote set-url origin https://github.com/YOUR-USERNAME/dimabsa-2026.git
```

## Best Practices

1. **Commit Messages**: Use clear, descriptive messages
2. **Branches**: Use branches for experiments
3. **Documentation**: Keep README updated
4. **Releases**: Tag important milestones
5. **Issues**: Use GitHub Issues for tracking

## Quick Commands Reference

```bash
# Check status
git status

# Add files
git add .

# Commit
git commit -m "Your message"

# Push
git push

# Pull latest
git pull

# Create branch
git checkout -b experiment-name

# Switch branch
git checkout main

# View history
git log --oneline
```

---

## Next Steps After GitHub Setup

1. ✅ Repository on GitHub
2. Clone in Kaggle or use GitHub integration
3. Run training
4. Push trained model (if using Git LFS)
5. Create release with results

---

*GitHub setup guide created: 2026-01-18*
