#!/bin/bash
# Script to initialize and push to GitHub

echo "=========================================="
echo "GitHub Repository Setup"
echo "=========================================="

# Set variables
REPO_DIR="/Users/vishal.thenuwara/Documents/MSC/Research/Coding/Competition/DimABSANew"
REPO_NAME="dimabsa-2026"

cd "$REPO_DIR"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
else
    echo "Git repository already initialized"
fi

# Check data file sizes
echo ""
echo "Checking data file sizes..."
du -h DESS/Codebase/data/dimabsa_combined/*.json

echo ""
echo "✅ Data files are small enough for GitHub (< 100 MB)"

# Add all files
echo ""
echo "Adding files to git..."
git add .

# Check status
echo ""
echo "Git status:"
git status --short | head -20
echo "..."

# Create initial commit
echo ""
read -p "Create initial commit? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git commit -m "Initial commit: DimABSA 2026 - DESS with VA Regression

- Phase 1: Data conversion (subtask_1, restaurant + laptop)
- Phase 2: Model modifications (VA regression)  
- Phase 3: Kaggle training setup
- All tests passing (8/8)
- 3,727 training samples ready
- Combined dataset: 8.3 MB training + 556 KB test"
    
    echo ""
    echo "✅ Initial commit created"
fi

# Instructions for GitHub
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Create repository on GitHub:"
echo "   https://github.com/new"
echo "   Name: $REPO_NAME"
echo "   Description: DimABSA 2026 - DESS Model with VA Regression"
echo ""
echo "2. Connect and push:"
echo "   git remote add origin https://github.com/YOUR-USERNAME/$REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Or use GitHub CLI:"
echo "   gh repo create $REPO_NAME --public --source=. --remote=origin --push"
echo ""
echo "=========================================="
echo "Repository ready for GitHub!"
echo "=========================================="
