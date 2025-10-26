# Lab 02: Making Your Project Functional with Data and Experiment Tracking

## Overview

Welcome to Lab 02! In this lab, you'll transform your well-structured ML project from Lab 01 into a fully functional, production-ready system with proper environment management, command-line interfaces, data versioning, and experiment tracking.

In Lab 01, you created a clean project structure and modularized your code. Now it's time to add the tools and practices that make your project:
- **Reproducible** - Anyone can recreate your environment and results
- **Accessible** - Command-line interfaces make your code easy to use
- **Trackable** - Version control for data and experiments
- **Testable** - Automated tests ensure code quality
- **Production-ready** - Ready for deployment and collaboration

### Learning Objectives

By the end of this lab, you will:

1. **Environment Management**: Understand computational environments and create isolated Python virtual environments
2. **Dependency Management**: Create and manage `requirements.txt` files
3. **CLI Development**: Build command-line interfaces using argparse
4. **Best Practices**: Implement hyperparameter tuning and scikit-learn pipelines
5. **Configuration Management**: Use YAML files for flexible configuration
6. **Data Versioning**: Use DVC (Data Version Control) with DagsHub
7. **Experiment Tracking**: Use MLflow to track experiments and model performance
8. **Testing**: Write unit tests with pytest

### Prerequisites

- Completed Lab 01 (modular project structure)
- Python 3.12.x installed
- Git repository set up
- Basic understanding of command line

---

## Part 1: Understanding Computational Environments

### What is a Computational Environment?

A **computational environment** is the complete software ecosystem where your code runs, including:
- Operating system
- Python interpreter version
- Installed packages and their versions
- System libraries
- Environment variables

### The "It Works on My Machine" Problem

Consider this scenario:
```
You: "My code works perfectly!"
Teammate: "It crashes on my machine..."
You: "But it works for me! 🤔"
```

**Why does this happen?**
- Different Python versions (3.10 vs 3.12)
- Different package versions (numpy 1.24 vs 2.3)
- Missing dependencies
- Different operating systems

### Why Standardized Environments Matter

**In Development:**
- Ensures code works identically for all team members
- Prevents "dependency hell"
- Makes onboarding new developers easier
- Enables reproducible research

**In Production:**
- Guarantees consistent behavior in deployment
- Enables rolling back to previous versions
- Facilitates automated testing and CI/CD
- Prevents production failures due to environment mismatches

### The Solution: Containerization and Virtual Environments

**Long-term solution (coming in Lab 04): Docker**
- Containers package your code AND its entire environment
- Works identically everywhere (local, cloud, different OS)
- Industry standard for deployment

**Today's solution: Python Virtual Environments**
- Isolated Python environment for your project
- Project-specific package versions
- Prevents conflicts between projects
- Lightweight and easy to use

**Why start with virtual environments?**
Our project currently uses only Python and doesn't require system-level dependencies. A Python virtual environment is sufficient for now. Later, when we deploy to production or need to ensure OS-level consistency, we'll containerize with Docker.

---

## Part 2: Creating and Managing Virtual Environments

### What is a Virtual Environment?

A Python virtual environment is an isolated Python installation that:
- Has its own Python interpreter copy
- Has its own `site-packages` directory (where packages install)
- Doesn't interfere with system Python or other projects
- Can have different package versions per project

**Analogy:** Think of it as a separate apartment for each project - each has its own furniture (packages) and doesn't share with others.

### Creating a Virtual Environment

Python includes the `venv` module for creating virtual environments:

```bash
# Create a virtual environment named .venv
python -m venv .venv
```

**Breaking down the command:**
- `python` - Run Python interpreter
- `-m venv` - Run the venv module as a script
- `.venv` - Name of the directory to create

**Why `.venv` (with a dot)?**
- Hidden directory (dot prefix on Unix/Linux/Mac)
- Industry convention
- Most IDEs auto-detect `.venv`
- Clearly indicates it's a virtual environment

**What gets created?**
```
.venv/
├── bin/              # Executables (Mac/Linux)
│   ├── python        # Python interpreter copy
│   ├── pip           # Package installer
│   └── activate      # Activation script
├── Scripts/          # Executables (Windows)
│   ├── python.exe
│   ├── pip.exe
│   └── activate.bat
├── lib/              # Installed packages
│   └── python3.12/
│       └── site-packages/
└── pyvenv.cfg        # Configuration
```

### Activating the Virtual Environment

**On Mac/Linux:**
```bash
source .venv/bin/activate
```

**On Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**How to tell it's activated?**
Your command prompt changes:
```bash
# Before activation
user@computer:~/project$

# After activation
(.venv) user@computer:~/project$
```

The `(.venv)` prefix indicates you're in the virtual environment!

### Verify Activation

```bash
# Check which Python is being used (should point to .venv)
which python    # Mac/Linux
where python    # Windows

# Output should be something like:
# /path/to/your/project/.venv/bin/python

# Check Python version
python --version
# Python 3.12.12

# List installed packages (should be minimal at first)
pip list
# Package    Version
# ---------- -------
# pip        24.0
# setuptools 65.5.0
```

### Deactivating the Virtual Environment

When you're done working:
```bash
deactivate
```

Your prompt returns to normal:
```bash
# After deactivation
user@computer:~/project$
```

### Best Practices

✅ **DO:**
- Create one virtual environment per project
- Name it `.venv` (convention)
- Activate before installing packages
- Commit requirements.txt, NOT the .venv folder
- Document activation steps in README

❌ **DON'T:**
- Commit the `.venv` folder to Git
- Share virtual environments between projects
- Install packages globally
- Forget to activate before working

### Adding .venv to .gitignore

Ensure your `.gitignore` includes:
```gitignore
# Virtual environments
.venv/
venv/
env/
ENV/
```

This prevents the (large!) virtual environment from being committed to Git.

---

## Part 3: Dependency Management with requirements.txt

### Why Document Dependencies?

Imagine you want to share your project:
```python
# Your code
import pandas as pd
import sklearn
import some_package_you_installed_6_months_ago
```

**Without documentation:**
- Users don't know what packages to install
- Users don't know which versions you used
- Code may break with different versions

**With requirements.txt:**
- Clear list of all dependencies
- Exact or compatible versions specified
- One command installs everything
- Reproducible environment

### The Two Approaches

#### Approach 1: pip freeze (Automatic)

```bash
# Activate your virtual environment
source .venv/bin/activate

# Generate requirements.txt with all installed packages
pip freeze > requirements.txt
```

**Example output:**
```txt
catboost==1.2.8
certifi==2025.1.12
charset-normalizer==3.4.1
contourpy==1.3.1
cycler==0.12.1
dvc==3.63.0
fonttools==4.55.3
idna==3.10
joblib==1.5.2
kiwisolver==1.4.7
...many more...
threadpoolctl==3.6.0
tzdata==2025.2
urllib3==2.3.0
```

**Pros:**
- Quick and automatic
- Captures exact versions
- Includes all dependencies

**Cons:**
- Lists sub-dependencies (packages you didn't directly install)
- Hard to read and understand
- Can cause conflicts
- Difficult to maintain

#### Approach 2: Hand-Curated (Recommended)

Manually create `requirements.txt` with only direct dependencies:

```txt
# Core ML and Data Science
numpy==2.3.4
pandas==2.3.3
scikit-learn==1.7.2

# Model Persistence
joblib==1.5.2

# Visualization
matplotlib==3.10.7
seaborn==0.13.2

# Testing
pytest==8.4.2
pytest-cov==7.0.0
```

**Pros:**
- Clear what YOU actually need
- Easier to read and maintain
- Sub-dependencies install automatically
- Can add comments
- More flexible

**Cons:**
- Requires manual maintenance
- Need to know what to include

### Middle Ground: Hybrid Approach (What We'll Use)

**Our recommendation:**
1. Use `pip freeze` to see everything installed
2. Manually create `requirements.txt` with only packages YOU installed
3. Remove sub-dependencies (they'll install automatically)

**Example workflow:**
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install packages you need
pip install numpy pandas scikit-learn joblib

# 3. See what got installed (including dependencies)
pip freeze
# numpy==2.3.4
# pandas==2.3.3
# python-dateutil==2.9.0.post0  # <- dependency of pandas
# pytz==2025.2                   # <- dependency of pandas
# scikit-learn==1.7.2
# scipy==1.16.2                  # <- dependency of scikit-learn
# joblib==1.5.2
# ...

# 4. Create requirements.txt with ONLY what YOU installed
nano requirements.txt
```

**Your `requirements.txt` should contain:**
```txt
numpy==2.3.4
pandas==2.3.3
scikit-learn==1.7.2
joblib==1.5.2
```

When someone runs `pip install -r requirements.txt`, pip automatically installs the dependencies (python-dateutil, pytz, scipy, etc.).

### Shell Redirection Explained

The `>` operator in bash is called **output redirection**:

```bash
# Instead of this (prints to screen):
pip freeze

# Do this (saves to file):
pip freeze > requirements.txt
```

**What it does:**
- Takes output that would go to terminal (stdout)
- Redirects it to a file instead
- Creates file if it doesn't exist
- **Overwrites** file if it does exist

**Related operators:**
```bash
# Overwrite file
pip freeze > requirements.txt

# Append to file
echo "# Additional comment" >> requirements.txt

# Show errors only
pip install 2> errors.txt
```

### Version Pinning Strategies

When specifying package versions, you have options:

#### 1. Exact Pinning (Most Restrictive)
```txt
numpy==2.3.4
```
**Meaning:** Install exactly version 2.3.4, nothing else

**Pros:**
- Perfectly reproducible
- No surprises

**Cons:**
- Misses bug fixes
- May conflict with other packages
- Requires manual updates

**Use when:** Production deployments, critical applications

#### 2. Compatible Release (Recommended for Development)
```txt
numpy>=2.3.0,<3.0.0
```
**Meaning:** Install version 2.3.0 or higher, but below 3.0.0

**Pros:**
- Gets bug fixes and minor updates
- Won't break on major version changes
- Good balance of stability and flexibility

**Cons:**
- Slightly less reproducible
- Small risk of breaking changes in minor versions

**Use when:** Development, educational projects (like this course)

#### 3. Minimum Version (Most Flexible)
```txt
numpy>=2.3.0
```
**Meaning:** Install version 2.3.0 or any newer version

**Pros:**
- Most flexible
- Always gets latest features

**Cons:**
- May break with new major versions
- Least reproducible

**Use when:** Rapid development, when you always want latest

#### 4. Compatible Version (~=)
```txt
numpy~=2.3.0
```
**Meaning:** Equivalent to `>=2.3.0,<2.4.0`

Gets patch releases (2.3.1, 2.3.2) but not minor releases (2.4.0)

### Our Requirements.txt for This Lab

**For Lab 02, we'll use exact pinning** to ensure everyone has the same environment:

```txt
# Core ML and Data Science
numpy==2.3.4
pandas==2.3.3
scikit-learn==1.7.2

# Model Persistence
joblib==1.5.2

# Visualization
matplotlib==3.10.7
seaborn==0.13.2
plotly==6.3.1
missingno==0.5.2

# Advanced ML Models
xgboost==3.1.1
catboost==1.2.8

# Configuration
PyYAML==6.0.3

# Data Version Control
dvc==3.63.0

# Experiment Tracking
mlflow==3.5.1

# Testing
pytest==8.4.2
pytest-cov==7.0.0

# Jupyter Notebooks
jupyter==1.1.1
```

### Installing from requirements.txt

Once you have a `requirements.txt`:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install all packages
pip install -r requirements.txt
```

**What happens:**
1. pip reads each line in requirements.txt
2. Downloads each package (and dependencies)
3. Installs in your virtual environment
4. Verifies all dependencies are compatible

**Verify installation:**
```bash
pip list

# Should show all packages from requirements.txt plus their dependencies
```

### Python Version Compatibility Note

⚠️ **Important:** At the time of writing, Python 3.13 is available but some packages (especially PyTorch, TensorFlow) don't fully support it yet. 

**Recommendation:** Use Python 3.12.x for maximum compatibility.

**Check your Python version:**
```bash
python --version
# Python 3.12.12 ✅ Good!
# Python 3.13.0 ⚠️ May have issues with some packages
```

If you have Python 3.13 and encounter issues, install Python 3.12 and recreate your virtual environment:
```bash
# On Mac (with Homebrew)
brew install python@3.12

# Create venv with specific Python version
python3.12 -m venv .venv

# Activate and install packages
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Part 4: Command-Line Interfaces with argparse

### Why Command-Line Interfaces (CLI)?

**Without CLI:**
```python
# Have to edit the file every time
def train_model():
    data_path = "data/processed/data.npy"  # Hard-coded
    model_type = "random_forest"           # Hard-coded
    tune = False                           # Hard-coded
    # ...
    
# Run script
python train.py
```

Every change requires editing the Python file!

**With CLI:**
```bash
# Use command-line arguments
python train.py --data data/processed/data.npy --model random_forest --tune

# Easy to change without editing code
python train.py --data data/other_data.npy --model gradient_boosting --tune

# Can be automated in scripts
./run_experiments.sh
```

**Benefits:**
- No code editing required
- Easy to automate
- Standard practice in industry
- Enables batch processing
- Better for production

### Introduction to argparse

Python's `argparse` module makes it easy to create CLIs:

```python
import argparse

def main():
    # Create parser
    parser = argparse.ArgumentParser(
        description='Train machine learning models'
    )
    
    # Add arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--model', type=str, default='random_forest',
                       help='Model type to train')
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use arguments
    print(f"Training {args.model} on {args.data}")
    if args.tune:
        print("Hyperparameter tuning enabled")

if __name__ == '__main__':
    main()
```

**Usage:**
```bash
# Get help
python train.py --help

# Run with arguments
python train.py --data mydata.npy --model random_forest --tune
```

### Argument Types

#### 1. Required Arguments
```python
parser.add_argument('--data', type=str, required=True,
                   help='Path to training data')
```
**Usage:** `--data file.npy` (must provide)

#### 2. Optional Arguments with Defaults
```python
parser.add_argument('--model', type=str, default='random_forest',
                   help='Model type')
```
**Usage:** 
- `--model decision_tree` (override default)
- Omit for default value

#### 3. Boolean Flags
```python
parser.add_argument('--tune', action='store_true',
                   help='Enable tuning')
```
**Usage:**
- Include `--tune` for True
- Omit for False

#### 4. Choices (Limited Options)
```python
parser.add_argument('--model', type=str,
                   choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
                   help='Model type')
```
**Usage:** Must be one of the listed choices

#### 5. Multiple Values
```python
parser.add_argument('--features', nargs='+',
                   help='Feature names')
```
**Usage:** `--features feature1 feature2 feature3`

### Example: Training Script with CLI

**src/train.py with argparse:**

```python
import argparse
from src.train import train_random_forest, save_model
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description='Train machine learning models for telecom churn prediction'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to preprocessed training data (numpy file)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['logistic_regression', 'random_forest', 'decision_tree',
                'adaboost', 'gradient_boosting', 'voting_classifier', 'all'],
        default='all',
        help='Model type to train (default: all)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning using GridSearchCV'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/',
        help='Directory to save trained models'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}")
    data = np.load(args.data, allow_pickle=True).item()
    X_train = data['X_train']
    y_train = data['y_train']
    
    # Train model
    if args.model == 'all':
        # Train all models
        models = train_all_models(X_train, y_train, tune_hyperparameters=args.tune)
        saved_paths = save_all_models(models, args.output_dir)
        print(f"\nTrained and saved {len(saved_paths)} models")
    else:
        # Train single model
        model = train_random_forest(X_train, y_train, tune_hyperparameters=args.tune)
        model_path = save_model(model, args.model, args.output_dir)
        print(f"\nTrained and saved {args.model}: {model_path}")

if __name__ == '__main__':
    main()
```

**Usage examples:**
```bash
# Train all models without tuning (fast)
python -m src.train --data data/processed/preprocessed_data.npy

# Train random forest with hyperparameter tuning
python -m src.train --data data/processed/preprocessed_data.npy --model random_forest --tune

# Train gradient boosting to custom directory
python -m src.train --data data/processed/preprocessed_data.npy --model gradient_boosting --output-dir my_models/

# Get help
python -m src.train --help
```

### Automatic Help Messages

argparse automatically generates help:

```bash
python -m src.train --help
```

**Output:**
```
usage: train.py [-h] --data DATA [--model {logistic_regression,random_forest,...}]
                [--tune] [--output-dir OUTPUT_DIR]

Train machine learning models for telecom churn prediction

options:
  -h, --help            show this help message and exit
  --data DATA           Path to preprocessed training data (numpy file)
  --model {logistic_regression,random_forest,...}
                        Model type to train (default: all)
  --tune                Enable hyperparameter tuning using GridSearchCV
  --output-dir OUTPUT_DIR
                        Directory to save trained models
```

### Best Practices for CLI Design

✅ **DO:**
- Provide sensible defaults
- Use `--help` text for every argument
- Use descriptive argument names
- Provide examples in docstrings
- Use choices for limited options
- Make required arguments explicit

❌ **DON'T:**
- Use single-letter arguments without long form
- Forget to validate inputs
- Hard-code values that could be arguments
- Skip the description parameter

---

## Part 5: Enhancements to Our Codebase

In this section, we'll improve our code with two important enhancements that were missing from the original Proof of Concept:

1. **Hyperparameter Tuning** - Finding optimal model parameters
2. **Scikit-learn Pipelines** - Professional preprocessing approach

### Why These Improvements?

**Original PoC Issues:**
- Used default hyperparameters (not optimal)
- Manual preprocessing (error-prone, not reproducible)
- No systematic way to find best parameters
- Preprocessing steps scattered across code

**After Improvements:**
- Automated hyperparameter search
- Reproducible preprocessing pipelines
- Better model performance
- Production-ready code

---

### Enhancement 1: Hyperparameter Tuning

#### What are Hyperparameters?

**Parameters** vs **Hyperparameters:**

**Parameters** (learned during training):
```python
# Linear regression: y = mx + b
# m and b are parameters learned from data
```

**Hyperparameters** (set before training):
```python
RandomForestClassifier(
    n_estimators=100,      # How many trees?
    max_depth=10,          # How deep each tree?
    min_samples_split=2,   # Min samples to split node?
    max_features='sqrt'    # Features per split?
)
```

These values affect how the model learns but aren't learned from data.

#### Why Tune Hyperparameters?

**Default values:**
```python
# Using defaults
model = RandomForestClassifier()  # Uses default hyperparameters
model.fit(X_train, y_train)
# Accuracy: 78%
```

**Tuned values:**
```python
# After tuning
model = RandomForestClassifier(
    n_estimators=200,        # Found through tuning
    max_depth=15,            # Found through tuning
    min_samples_split=5      # Found through tuning
)
model.fit(X_train, y_train)
# Accuracy: 82%  🎉 4% improvement!
```

**Default parameters are rarely optimal** for your specific dataset!

#### Grid Search: Systematic Hyperparameter Tuning

**Grid Search** tries all combinations of hyperparameters:

```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],           # 3 options
    'max_depth': [10, 20, 30],                # 3 options
    'min_samples_split': [2, 5, 10]           # 3 options
}
# Total combinations: 3 × 3 × 3 = 27 models to try

# Create base model
base_model = RandomForestClassifier(random_state=42)

# Create grid search
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,              # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,         # Use all CPU cores
    verbose=1          # Show progress
)

# Fit (tries all combinations)
grid_search.fit(X_train, y_train)

# Best model found
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

**What happens:**
1. GridSearchCV creates 27 models (one for each combination)
2. For each model, performs 5-fold cross-validation
3. Total: 27 × 5 = 135 model training runs
4. Returns the model with best average performance

#### Cross-Validation Explained

**Simple train-test split:**
```
[Training Data ----------------] [Test Data ----]
                                 ↑ Single evaluation
```
**Problem:** Test set might not be representative

**5-Fold Cross-Validation:**
```
Fold 1: [Test] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Test] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Test] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Test] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Test]
        
Average the 5 scores → More reliable estimate
```

**Benefits:**
- Uses all data for both training and validation
- More reliable performance estimate
- Reduces overfitting to specific test set

#### Implementation in Our Code

**Before (no tuning):**
```python
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
```

**After (with optional tuning):**
```python
def train_random_forest(X_train, y_train, tune_hyperparameters=False):
    if tune_hyperparameters:
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Grid search with cross-validation
        base_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    else:
        # Use defaults (fast)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model
```

**Usage:**
```bash
# Without tuning (fast, uses defaults)
python -m src.train --data data.npy --model random_forest

# With tuning (slower, finds best parameters)
python -m src.train --data data.npy --model random_forest --tune
```

#### Trade-offs

**Without Tuning:**
- ⚡ Fast (seconds to minutes)
- 🎯 Acceptable performance
- 👍 Good for prototyping

**With Tuning:**
- 🐌 Slow (minutes to hours)
- 🎯 Better performance
- 👍 Worth it for final models

**Recommendation:** 
- Start without tuning during development
- Enable tuning for final models
- Use for models you'll deploy

---

### Enhancement 2: Scikit-learn Pipelines

#### The Problem with Manual Preprocessing

**Manual approach** (what we did in Lab 01):

```python
# Step 1: Scale training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 2: Scale test data
X_test_scaled = scaler.transform(X_test)

# Step 3: Train model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Step 4: Make predictions on new data
# ⚠️ PROBLEM: Need to remember to scale new data!
new_data_scaled = scaler.transform(new_data)  # Must remember this!
predictions = model.predict(new_data_scaled)
```

**Problems:**
1. **Easy to forget steps** - What if you forget to scale new data?
2. **Data leakage risk** - What if you fit scaler on test data by mistake?
3. **Not reproducible** - Hard to package preprocessing + model together
4. **Deployment issues** - Need to save scaler separately and remember to apply it

#### The Solution: Scikit-learn Pipelines

A **Pipeline** bundles preprocessing and model into one object:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),          # Step 1: Scale
    ('classifier', RandomForestClassifier())  # Step 2: Classify
])

# Train (automatically scales then trains)
pipeline.fit(X_train, y_train)

# Predict (automatically scales then predicts)
predictions = pipeline.predict(new_data)  # Scaling happens automatically!
```

**Benefits:**
✅ Can't forget preprocessing steps
✅ Prevents data leakage
✅ One object contains everything
✅ Easy to deploy
✅ Reproducible

#### ColumnTransformer: Different Preprocessing for Different Columns

Real datasets have different types of features:
- **Numerical features**: age, income, charges → Need scaling
- **Categorical features**: gender, contract type → Need encoding

**ColumnTransformer** applies different preprocessing to different columns:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Define which columns get which preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'income', 'charges']),
        ('cat', OrdinalEncoder(), ['gender', 'contract', 'payment_method'])
    ]
)

# Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train and predict (all preprocessing automatic!)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

#### Our Implementation

**New function in `preprocess.py`:**

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Create a scikit-learn preprocessing pipeline.
    
    This is the RECOMMENDED way to preprocess data in production as it:
    1. Ensures consistency between training and prediction
    2. Prevents data leakage
    3. Makes the pipeline reproducible and deployable
    4. Bundles all preprocessing steps together
    """
    # Numerical features: scale them
    numerical_transformer = StandardScaler()
    
    # Categorical features: encode them
    categorical_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    return pipeline
```

**Usage:**
```python
# Create pipeline
pipeline = create_preprocessing_pipeline(
    numerical_features=['tenure', 'MonthlyCharges', 'TotalCharges'],
    categorical_features=['gender', 'Contract', 'PaymentMethod']
)

# Fit on training data
X_train_transformed = pipeline.fit_transform(X_train)

# Transform test data (using fitted pipeline)
X_test_transformed = pipeline.transform(X_test)

# Save pipeline for later
joblib.dump(pipeline, 'preprocessing_pipeline.pkl')

# Load and use on new data
pipeline = joblib.load('preprocessing_pipeline.pkl')
new_data_transformed = pipeline.transform(new_data)
```

#### Backward Compatibility

We kept the old approach for backward compatibility:

```python
def preprocess_pipeline(filepath, use_sklearn_pipeline=True):
    # ... load and clean data ...
    
    if use_sklearn_pipeline:
        # New way (RECOMMENDED)
        X_train_transformed, pipeline = preprocess_data_with_pipeline(X_train, fit=True)
        X_test_transformed, _ = preprocess_data_with_pipeline(X_test, pipeline=pipeline, fit=False)
        return X_train_transformed, X_test_transformed, y_train, y_test, pipeline
    else:
        # Old way (LEGACY - still works but not recommended)
        # ... manual preprocessing ...
        return X_train, X_test, y_train, y_test, scaler
```

#### Why Pipelines Are Better

| Aspect | Manual Preprocessing | Scikit-learn Pipeline |
|--------|---------------------|----------------------|
| **Consistency** | ❌ Can forget steps | ✅ Automatic |
| **Data Leakage** | ⚠️ Easy to make mistakes | ✅ Prevented |
| **Deployment** | ❌ Complex | ✅ Simple |
| **Reproducibility** | ❌ Hard | ✅ Easy |
| **Code Organization** | ❌ Scattered | ✅ Centralized |

**Recommendation:** Always use pipelines in production code!

---

### Import Organization (PEP 8 Style)

Python's style guide (PEP 8) recommends organizing imports in this order:

```python
# 1. Standard library imports (built-in Python modules)
import argparse
import logging
import os
from datetime import datetime
from typing import Any, Dict

# 2. Related third-party imports (installed packages)
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 3. Local application imports (your own modules)
from .utils.config import RANDOM_STATE, MODELS_PATH
from .preprocess import load_data
```

**Within each group:**
- Separate `import` statements from `from ... import ...` statements
- Sort alphabetically
- Group related imports

**Why this matters:**
- Consistent style across the project
- Easier to find imports
- Prevents circular import issues
- Industry standard

**Example from our updated `train.py`:**
```python
"""
Model training module for telecom churn prediction.
"""

# Standard library imports
import argparse
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

# Third-party imports
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Local imports
from .utils.config import MODELS_PATH, RANDOM_STATE
```

---

## Part 6: YAML Configuration Files

### Why Move Beyond Python Config Files?

**Our current config (`config.py`):**
```python
# config.py
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

CATEGORICAL_FEATURES = ['gender', 'Contract', 'PaymentMethod']
NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges']
```

**Problems:**
- Must restart Python to reload changes
- Can't easily switch between configurations
- Harder for non-programmers to modify
- No comments inline with values

### YAML: Human-Friendly Configuration

YAML (YAML Ain't Markup Language) is a human-readable data format:

```yaml
# train_config.yaml
model:
  type: random_forest
  params:
    n_estimators: 100  # Number of trees
    max_depth: 10      # Maximum tree depth
    random_state: 42   # For reproducibility

training:
  test_size: 0.2       # 20% for testing
  cv_folds: 5          # Cross-validation folds

paths:
  data: data/processed/train_data.csv
  output: models/
```

**Benefits:**
✅ Easy to read and edit
✅ Can add comments
✅ Supports nested structures
✅ Standard for configuration
✅ Used by many tools (Docker, Kubernetes, DVC, MLflow)

### YAML Syntax Basics

#### Key-Value Pairs
```yaml
name: John Doe
age: 30
city: Edmonton
```

#### Nested Objects
```yaml
person:
  name: John Doe
  age: 30
  address:
    city: Edmonton
    country: Canada
```

#### Lists
```yaml
# Method 1: Dash notation
fruits:
  - apple
  - banana
  - orange

# Method 2: Inline notation
colors: [red, green, blue]
```

#### Comments
```yaml
# This is a comment
random_state: 42  # Inline comment
```

#### Data Types
```yaml
string: "Hello World"
integer: 42
float: 3.14
boolean: true
null_value: null
```

### Creating Configuration Files

**Create `configs/` directory:**
```bash
mkdir -p configs
```

**1. Training Configuration (`configs/train_config.yaml`):**

```yaml
# Training Configuration for Telecom Churn Prediction

# Model settings
model:
  type: random_forest  # Options: random_forest, gradient_boosting, logistic_regression
  random_state: 42     # For reproducibility
  
  # Hyperparameters (used if tune=false)
  params:
    random_forest:
      n_estimators: 100
      max_depth: 10
      min_samples_split: 2
      max_features: sqrt
    
    gradient_boosting:
      n_estimators: 100
      learning_rate: 0.1
      max_depth: 5
    
    logistic_regression:
      max_iter: 1000
      C: 1.0
      solver: lbfgs

# Training settings
training:
  test_size: 0.2       # Proportion for test set
  cv_folds: 5          # Cross-validation folds
  tune: false          # Enable hyperparameter tuning
  n_jobs: -1           # CPU cores (-1 = use all)

# Hyperparameter tuning grid (used if tune=true)
tuning:
  random_forest:
    n_estimators: [50, 100, 200]
    max_depth: [10, 20, 30, null]
    min_samples_split: [2, 5, 10]
    max_features: [sqrt, log2]
  
  gradient_boosting:
    n_estimators: [50, 100, 200]
    learning_rate: [0.01, 0.1, 0.2]
    max_depth: [3, 5, 7]

# Paths
paths:
  data:
    raw: data/raw/
    processed: data/processed/
  models: models/
  outputs: outputs/

# Logging
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

**2. Preprocessing Configuration (`configs/preprocess_config.yaml`):**

```yaml
# Preprocessing Configuration for Telecom Churn Prediction

# Data loading
data:
  filename: WA_Fn-UseC_-Telco-Customer-Churn.csv
  target_column: Churn
  id_columns: [customerID]

# Missing value handling
missing_values:
  strategy: fill  # Options: fill, drop
  fill_method: mean  # For numerical: mean, median, mode
  threshold: 0.5  # Drop columns with >50% missing

# Feature encoding
encoding:
  categorical_method: ordinal  # Options: ordinal, onehot
  handle_unknown: use_encoded_value  # For new categories

# Scaling
scaling:
  method: standard  # Options: standard, minmax, robust
  with_mean: true
  with_std: true

# Feature columns
features:
  categorical:
    - gender
    - Partner
    - Dependents
    - PhoneService
    - MultipleLines
    - InternetService
    - OnlineSecurity
    - OnlineBackup
    - DeviceProtection
    - TechSupport
    - StreamingTV
    - StreamingMovies
    - Contract
    - PaperlessBilling
    - PaymentMethod
  
  numerical:
    - SeniorCitizen
    - tenure
    - MonthlyCharges
    - TotalCharges

# Train-test split
split:
  test_size: 0.2
  random_state: 42
  stratify: true  # Stratify by target

# Pipeline options
pipeline:
  use_sklearn_pipeline: true  # Recommended: true
  save_artifacts: true  # Save pipeline and encoders
```

**3. Prediction Configuration (`configs/predict_config.yaml`):**

```yaml
# Prediction Configuration

# Model
model:
  path: models/random_forest_20241027_143022.pkl
  type: random_forest

# Preprocessing
preprocessing:
  pipeline_path: data/processed/preprocessing_pipeline.pkl
  label_encoder_path: data/processed/label_encoder.pkl

# Input data
input:
  format: csv  # Options: csv, json, numpy
  path: data/new_data.csv

# Output
output:
  format: csv  # Options: csv, json, numpy
  path: predictions/predictions.csv
  include_probabilities: true
  include_confidence: true

# Batch processing
batch:
  enabled: true
  batch_size: 1000  # Process 1000 samples at a time
```

### Loading YAML in Python

**Install PyYAML:**
```bash
pip install PyYAML
```

**Load configuration:**
```python
import yaml

def load_config(config_path):
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Usage
config = load_config('configs/train_config.yaml')

# Access nested values
random_state = config['model']['random_state']
n_estimators = config['model']['params']['random_forest']['n_estimators']
test_size = config['training']['test_size']

print(f"Random state: {random_state}")
print(f"N estimators: {n_estimators}")
print(f"Test size: {test_size}")
```

### Using Configuration in Code

**Example: Training with YAML config**

```python
# train.py
import yaml
from src.train import train_random_forest, save_model

def main():
    # Load configuration
    with open('configs/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract settings
    model_type = config['model']['type']
    model_params = config['model']['params'][model_type]
    random_state = config['model']['random_state']
    tune = config['training']['tune']
    
    # Load data
    data_path = config['paths']['data']['processed'] + 'train_data.npy'
    data = np.load(data_path, allow_pickle=True).item()
    X_train = data['X_train']
    y_train = data['y_train']
    
    # Train model with config parameters
    model = train_random_forest(
        X_train,
        y_train,
        tune_hyperparameters=tune,
        **model_params
    )
    
    # Save model
    output_dir = config['paths']['models']
    save_model(model, model_type, output_dir)
    
    print(f"Model trained and saved using config: {config_path}")

if __name__ == '__main__':
    main()
```

### Combining CLI and YAML

**Best practice:** Use YAML for defaults, CLI for overrides

```python
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str,
                       default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--tune', action='store_true',
                       help='Override config: enable tuning')
    parser.add_argument('--model', type=str,
                       help='Override config: model type')
    
    args = parser.parse_args()
    
    # Load YAML config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with CLI arguments if provided
    if args.tune:
        config['training']['tune'] = True
    if args.model:
        config['model']['type'] = args.model
    
    # Use merged configuration
    # ...
```

**Usage:**
```bash
# Use defaults from YAML
python train.py

# Override specific settings
python train.py --tune
python train.py --model gradient_boosting --tune
python train.py --config configs/experimental_config.yaml
```

### Multiple Configurations for Different Scenarios

Create different configs for different use cases:

```bash
configs/
├── train_config.yaml          # Default training
├── train_config_fast.yaml     # Quick experiments (no tuning)
├── train_config_production.yaml  # Production (with tuning)
├── preprocess_config.yaml     # Default preprocessing
└── predict_config.yaml        # Prediction settings
```

**Example usage:**
```bash
# Fast experimentation
python train.py --config configs/train_config_fast.yaml

# Production training
python train.py --config configs/train_config_production.yaml
```

### Benefits of YAML Configuration

| Aspect | Hard-coded / config.py | YAML Configuration |
|--------|----------------------|-------------------|
| **Readability** | ❌ Python syntax | ✅ Plain text |
| **Comments** | ✅ Yes | ✅ Yes, inline |
| **Nested Data** | ⚠️ Verbose | ✅ Clean |
| **Hot Reload** | ❌ Restart needed | ✅ Just reload file |
| **Non-programmers** | ❌ Hard to edit | ✅ Easy to edit |
| **Version Control** | ✅ Yes | ✅ Yes |
| **Multiple Configs** | ❌ Complex | ✅ Easy |

---

## Summary of Part 1-6

Congratulations! You've now:

✅ **Understood computational environments** and why they matter
✅ **Created virtual environments** for isolated Python installations
✅ **Managed dependencies** with `requirements.txt`
✅ **Built CLI interfaces** using argparse
✅ **Added hyperparameter tuning** to improve models
✅ **Implemented scikit-learn pipelines** for better preprocessing
✅ **Organized imports** following PEP 8 style
✅ **Created YAML configurations** for flexible settings

### Your Updated Project Structure

```
your-project/
├── .venv/                      # Virtual environment (not in Git)
├── configs/                    # YAML configuration files
│   ├── train_config.yaml
│   ├── preprocess_config.yaml
│   └── predict_config.yaml
├── data/
│   ├── raw/
│   └── processed/
│       ├── preprocessed_data.npy
│       ├── preprocessing_pipeline.pkl  # NEW: Saved pipeline
│       └── label_encoder.pkl           # NEW: Saved encoder
├── src/
│   ├── preprocess.py           # UPDATED: With sklearn pipelines & CLI
│   ├── train.py                # UPDATED: With tuning & CLI
│   ├── predict.py              # UPDATED: With CLI
│   ├── evaluate.py             # UPDATED: With CLI
│   └── utils/
│       └── config.py
├── models/                     # Saved models
├── requirements.txt            # NEW: Dependencies
└── README.md
```

### Quick Command Reference

```bash
# Virtual Environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Install Dependencies
pip install -r requirements.txt

# Preprocessing (with pipeline)
python -m src.preprocess --input data/raw/data.csv

# Training (with tuning)
python -m src.train --data data/processed/preprocessed_data.npy --model random_forest --tune

# Prediction
python -m src.predict --model models/model.pkl --data data/processed/preprocessed_data.npy

# Evaluation
python -m src.evaluate --model models/model.pkl --data data/processed/preprocessed_data.npy
```

---

## Next: DVC, MLflow, and Testing

In the next sections of Lab 02, we'll cover:
- **DVC (Data Version Control)** with DagsHub
- **MLflow** for experiment tracking
- **pytest** for automated testing

Stay tuned! 🚀

---

*Lab 02 Instructions - Part 1 of 2*  
*CMPT 2500 - ML/AI Deployment*  
*NorQuest College*
