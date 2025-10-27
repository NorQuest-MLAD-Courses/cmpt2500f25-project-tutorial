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

```text
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

**In Development**:

- Ensures code works identically for all team members
- Prevents "dependency hell"
- Makes onboarding new developers easier
- Enables reproducible research

**In Production**:

- Guarantees consistent behavior in deployment
- Enables rolling back to previous versions
- Facilitates automated testing and CI/CD
- Prevents production failures due to environment mismatches

### The Solution: Containerization and Virtual Environments

**Long-term solution** (coming in Lab 04): **Docker**

- Containers package your code AND its entire environment
- Works identically everywhere (local, cloud, different OS)
- Industry standard for deployment

**Today's solution**: Python **Virtual Environments**

- Isolated Python environment for your project
- Project-specific package versions
- Prevents conflicts between projects
- Lightweight and easy to use

**Why start with virtual environments**?
Our project currently uses only Python and doesn't require system-level dependencies. A Python virtual environment is sufficient for now. Later, when we deploy to production or need to ensure OS-level consistency, we'll containerize with Docker.

---

## Part 2: Creating and Managing Virtual Environments

### What is a Virtual Environment?

A Python virtual environment is an isolated Python installation that:

- Has its own Python interpreter copy
- Has its own `site-packages` directory (where packages install)
- Doesn't interfere with system Python or other projects
- Can have different package versions per project

**Analogy**: Think of it as a separate apartment for each project - each has its own furniture (packages) and doesn't share with others.

### Creating a Virtual Environment

Python includes the `venv` module for creating virtual environments:

```bash
# Create a virtual environment named .venv
python -m venv .venv
```

**Breaking down the command**:

- `python` - Run Python interpreter
- `-m venv` - Run the venv module as a script
- `.venv` - Name of the directory to create

**Why `.venv` (with a dot)?**

- Hidden directory (dot prefix on Unix/Linux/Mac)
- Industry convention
- Most IDEs auto-detect `.venv`
- Clearly indicates it's a virtual environment

**What gets created?**

```output
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

**On Mac/Linux**:

```bash
source .venv/bin/activate
```

**On Windows (Command Prompt)**:

```cmd
.venv\Scripts\activate.bat
```

**On Windows (PowerShell)**:

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

✅ **DO**:

- Create one virtual environment per project
- Name it `.venv` (convention)
- Activate before installing packages
- Commit requirements.txt, NOT the .venv folder
- Document activation steps in README

❌ **DON'T**:

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

**Without documentation**:

- Users don't know what packages to install
- Users don't know which versions you used
- Code may break with different versions

**With requirements.txt**:

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

**Example output**:

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

**Pros**:

- Quick and automatic
- Captures exact versions
- Includes all dependencies

**Cons**:

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
plotly==6.3.1

# Advanced Models
xgboost==3.1.1
catboost==1.2.8

# Configuration
PyYAML==6.0.3

# Data Version Control (DagsHub remote - recommended)
dvc==3.63.0
dagshub==0.6.3

# Testing
pytest==8.4.2
pytest-cov==7.0.0

# Development
jupyter==1.1.1
```

**Pros**:

- Clean and readable
- Only direct dependencies
- Easy to maintain
- Documents what you actually use
- Less likely to cause conflicts

**Cons**:

- Requires manual maintenance
- Need to remember to update when adding packages

### Version Pinning Strategies

**Exact pinning (==)**:

```txt
pandas==2.3.3  # Exact version
```

- Most reproducible
- Safest for production
- May miss bug fixes

**Compatible release (~=)**:

```txt
pandas~=2.3.3  # Compatible: >=2.3.3, <2.4.0
```

- Allows patch updates
- Balance of stability and updates

**Minimum version (>=)**:

```txt
pandas>=2.3.0  # Any version 2.3.0 or higher
```

- Most flexible
- Can break compatibility
- Use with caution

**Recommended approach**:

```txt
# Production code - exact pinning
pandas==2.3.3

# Library development - compatible release
pandas~=2.3.3

# Quick experiments - minimum version
pandas>=2.3.0
```

### Installing from requirements.txt

```bash
# Activate virtual environment first!
source .venv/bin/activate

# Install all requirements
pip install -r requirements.txt

# Upgrade pip first (recommended)
pip install --upgrade pip
pip install -r requirements.txt

# Force reinstall if needed
pip install --force-reinstall -r requirements.txt
```

### Best Practices for requirements.txt

✅ **DO**:

- Use hand-curated approach
- Group packages logically with comments
- Pin exact versions for production
- Update when adding new packages
- Test installation on clean environment

❌ **DON'T**:

- Use `pip freeze` blindly
- Include OS-specific packages
- Commit without testing
- Use vague version specifiers in production

---

## Part 4: Command-Line Interfaces with argparse

### Why CLI Interfaces Matter

**Before CLI (hard-coded values)**:

```python
# train.py
def main():
    data_path = "data/processed/train.npy"  # Hard-coded!
    model_type = "random_forest"            # Hard-coded!
    tune = False                             # Hard-coded!
    
    # To change: Edit the file, save, run again
```

**After CLI (argparse)**:

```bash
# Flexible usage from command line
python -m src.train --data data/processed/train.npy --model random_forest --tune
python -m src.train --data data/processed/train.npy --model logistic_regression
python -m src.train --data other_data.npy --model gradient_boosting --tune
```

**Benefits**:

- ✅ No code changes needed for different runs
- ✅ Easy to automate (scripts, CI/CD)
- ✅ Standard interface everyone understands
- ✅ Self-documenting with `--help`
- ✅ Type checking and validation

### Introduction to argparse

Python's `argparse` module is the standard library tool for creating CLI interfaces.

**Basic example**:

```python
import argparse

def main():
    # Create parser
    parser = argparse.ArgumentParser(
        description='Train a machine learning model'
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
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Tune: {args.tune}")

if __name__ == '__main__':
    main()
```

**Usage**:

```bash
# Get help
python train.py --help

# Run with arguments
python train.py --data data/train.npy --model random_forest
python train.py --data data/train.npy --model logistic_regression --tune
```

### Argument Types

**1. Required arguments**:

```python
parser.add_argument('--data', type=str, required=True,
                   help='Path to training data')
```

**2. Optional arguments with defaults**:

```python
parser.add_argument('--model', type=str, default='random_forest',
                   help='Model type')
```

**3. Boolean flags**:

```python
parser.add_argument('--tune', action='store_true',
                   help='Enable tuning')
```

- Present → True
- Absent → False

**4. Choices (restricted values)**:

```python
parser.add_argument('--model', type=str,
                   choices=['random_forest', 'logistic_regression', 'decision_tree'],
                   help='Model type')
```

**5. Multiple values**:

```python
parser.add_argument('--features', nargs='+', type=str,
                   help='Feature columns')
```

### Complete Training CLI Example

```python
# src/train.py
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Train machine learning models for churn prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train Random Forest with default parameters
  python -m src.train --data data/processed/preprocessed_data.npy --model random_forest
  
  # Train with hyperparameter tuning
  python -m src.train --data data/processed/preprocessed_data.npy --model random_forest --tune
  
  # Train all models
  python -m src.train --data data/processed/preprocessed_data.npy --model all
        '''
    )
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to preprocessed training data (.npy file)')
    
    # Optional arguments
    parser.add_argument('--model', type=str, 
                       default='random_forest',
                       choices=['logistic_regression', 'random_forest', 
                               'decision_tree', 'adaboost', 
                               'gradient_boosting', 'voting_classifier', 'all'],
                       help='Model type to train (default: random_forest)')
    
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning with GridSearchCV')
    
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models (default: models)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Hyperparameter Tuning: {args.tune}")
    print(f"Output Directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Load data
    data = np.load(args.data, allow_pickle=True).item()
    X_train = data['X_train']
    y_train = data['y_train']
    
    print(f"Loaded data: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    
    # Train model(s)
    if args.model == 'all':
        # Train all models
        models = ['logistic_regression', 'random_forest', 'decision_tree',
                 'adaboost', 'gradient_boosting', 'voting_classifier']
        for model_type in models:
            train_and_save_model(model_type, X_train, y_train, args.tune, args.output_dir)
    else:
        train_and_save_model(args.model, X_train, y_train, args.tune, args.output_dir)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")

def train_and_save_model(model_type, X_train, y_train, tune, output_dir):
    """Train and save a single model."""
    print(f"\nTraining {model_type}...")
    start_time = datetime.now()
    
    # Train model (import your training functions)
    from src.train import (train_logistic_regression, train_random_forest,
                          train_decision_tree, train_adaboost,
                          train_gradient_boosting, train_voting_classifier)
    
    train_funcs = {
        'logistic_regression': train_logistic_regression,
        'random_forest': train_random_forest,
        'decision_tree': train_decision_tree,
        'adaboost': train_adaboost,
        'gradient_boosting': train_gradient_boosting,
        'voting_classifier': train_voting_classifier
    }
    
    model = train_funcs[model_type](X_train, y_train, tune_hyperparameters=tune)
    
    # Save model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{output_dir}/{model_type}_{timestamp}.pkl"
    
    import joblib
    joblib.dump(model, model_path)
    
    elapsed = datetime.now() - start_time
    print(f"✓ Model saved: {model_path}")
    print(f"  Training time: {elapsed.total_seconds():.2f}s")

if __name__ == '__main__':
    main()
```

### CLI Best Practices

✅ **DO**:

- Provide clear, descriptive help messages
- Use sensible defaults
- Include usage examples in epilog
- Validate input files/paths
- Print informative output
- Use `--help` flag

❌ **DON'T**:

- Make everything required (use defaults)
- Use cryptic argument names
- Skip help messages
- Forget to validate inputs
- Run silently without output

---

## Part 5: Code Enhancements

### Hyperparameter Tuning with GridSearchCV

**What is Hyperparameter Tuning?**

Hyperparameters are settings you choose before training (e.g., number of trees, learning rate). Unlike model parameters (learned during training), hyperparameters significantly affect model performance.

**Manual tuning**:

```python
# Try different values manually
model1 = RandomForestClassifier(n_estimators=50, max_depth=5)
model2 = RandomForestClassifier(n_estimators=100, max_depth=10)
model3 = RandomForestClassifier(n_estimators=200, max_depth=15)
# ... tedious and time-consuming
```

**GridSearchCV (automated)**:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create model
model = RandomForestClassifier(random_state=42)

# GridSearchCV tries all combinations with cross-validation
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,              # Use all CPU cores
    verbose=1
)

# Fit on data
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best CV score: {best_score:.4f}")
```

**Implementation in train.py**:

```python
def train_random_forest(X_train, y_train, tune_hyperparameters=False):
    """
    Train Random Forest Classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        tune_hyperparameters: If True, use GridSearchCV
        
    Returns:
        Trained model
    """
    if tune_hyperparameters:
        print("Training with hyperparameter tuning (this may take a while)...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
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
        print("Training with default hyperparameters...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
```

**Usage**:

```bash
# Fast: default hyperparameters (~30 seconds)
python -m src.train --data data/processed/preprocessed_data.npy --model random_forest

# Slow but better: tuned hyperparameters (~5-10 minutes)
python -m src.train --data data/processed/preprocessed_data.npy --model random_forest --tune
```

### Scikit-learn Pipelines for Preprocessing

**Problem with manual preprocessing**:

```python
# Training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)
joblib.dump(scaler, 'scaler.pkl')  # Save separately!
joblib.dump(model, 'model.pkl')    # Save separately!

# Prediction
scaler = joblib.load('scaler.pkl')  # Load both!
model = joblib.load('model.pkl')
X_new_scaled = scaler.transform(X_new)  # Don't forget to scale!
predictions = model.predict(X_new_scaled)
```

**Issues**:

- ❌ Easy to forget preprocessing step
- ❌ Two files to manage
- ❌ Can apply wrong scaler
- ❌ Hard to ensure consistency

**Solution**: Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Train (preprocessing happens automatically)
pipeline.fit(X_train, y_train)

# Save (saves entire pipeline as one object)
joblib.dump(pipeline, 'model_pipeline.pkl')

# Predict (preprocessing happens automatically)
pipeline = joblib.load('model_pipeline.pkl')
predictions = pipeline.predict(X_new)  # Scaling applied automatically!
```

**Benefits**:

- ✅ Preprocessing and model bundled together
- ✅ One file to save/load
- ✅ Automatic preprocessing during prediction
- ✅ Prevents preprocessing errors
- ✅ Cleaner, more maintainable code

**Advanced**: ColumnTransformer for Different Feature Types

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define feature groups
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['gender', 'Contract', 'PaymentMethod']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Complete pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Use it
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Implementation in preprocess.py**:

```python
def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Create sklearn preprocessing pipeline.
    
    Args:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        
    Returns:
        ColumnTransformer pipeline
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), 
             categorical_features)
        ],
        remainder='passthrough'  # Keep other columns as-is
    )
    
    return preprocessor

def main():
    # ... load data ...
    
    # Define feature groups
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService',
                           'MultipleLines', 'InternetService', 'OnlineSecurity',
                           'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod']
    
    # Create pipeline
    pipeline = create_preprocessing_pipeline(numerical_features, categorical_features)
    
    # Fit and transform
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    
    # Save pipeline
    joblib.dump(pipeline, 'data/processed/preprocessing_pipeline.pkl')
    
    # Save data
    data = {
        'X_train': X_train_transformed,
        'X_test': X_test_transformed,
        'y_train': y_train,
        'y_test': y_test
    }
    np.save('data/processed/preprocessed_data.npy', data)
```

### Import Organization (PEP 8)

**PEP 8** is Python's style guide. Proper import organization makes code more readable and professional.

**Bad (disorganized)**:

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sys
from datetime import datetime
import numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
import os
```

**Good (PEP 8 compliant)**:

```python
# Standard library imports
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Local imports
from src.utils.config import Config
from src.utils.helpers import load_data, save_model
```

**PEP 8 Import Order**:

1. **Standard library** (built into Python)
2. **Third-party packages** (installed via pip)
3. **Local application** (your own modules)

**Blank lines**:

- One blank line between groups
- Alphabetical order within groups (optional but recommended)

**Why it matters**:

- ✅ Immediately see what external dependencies exist
- ✅ Easier to identify missing imports
- ✅ Professional, maintainable code
- ✅ Follows Python community standards
- ✅ Better for code reviews

---

## Part 6: Configuration Management with YAML

### Why Use Configuration Files?

**Problem**: Hard-coded values scattered everywhere

```python
# train.py
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
test_size = 0.2
batch_size = 32

# preprocess.py
scaler_type = 'standard'
handle_missing = 'mean'

# predict.py
threshold = 0.5
```

**To change anything**: Edit multiple files, risk introducing bugs

**Solution**: Centralized Configuration

```yaml
# configs/train_config.yaml
model:
  type: random_forest
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42

training:
  test_size: 0.2
  batch_size: 32

preprocessing:
  scaler: standard
  missing_strategy: mean

prediction:
  threshold: 0.5
```

**To change anything**: Edit one YAML file, no code changes needed!

### YAML Syntax Basics

YAML (YAML Ain't Markup Language) is a human-readable data format.

**Key-value pairs**:

```yaml
name: John Doe
age: 30
active: true
```

**Nested structures**:

```yaml
person:
  name: John Doe
  age: 30
  address:
    street: 123 Main St
    city: New York
```

**Lists**:

```yaml
fruits:
  - apple
  - banana
  - orange

# Or inline
colors: [red, green, blue]
```

**Comments**:

```yaml
# This is a comment
name: John  # Inline comment
```

**Data types**:

```yaml
string: "Hello"
integer: 42
float: 3.14
boolean: true
null_value: null
```

### Example Configuration Files

**1. Main Configuration (`configs/config.yaml`)**:

```yaml
# Project Configuration

# Project metadata
project:
  name: cmpt2500f25-project-tutorial
  version: 2.0.0
  description: Tutorial project for CMPT 2500: Customer churn prediction for telecommunications

# Paths
paths:
  data:
    raw: data/raw/
    processed: data/processed/
    external: data/external/
  models: models/
  outputs: outputs/
  logs: logs/

# Random seed for reproducibility
random_state: 42

# Logging
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/app.log
```

**2. Training Configuration (`configs/train_config.yaml`)**:

```yaml
# Training Configuration

# Model selection
model:
  type: random_forest  # Options: logistic_regression, random_forest, decision_tree, etc.
  random_state: 42

# Model-specific parameters
params:
  logistic_regression:
    max_iter: 1000
    solver: lbfgs
    C: 1.0
  
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    min_samples_leaf: 1
  
  decision_tree:
    max_depth: 10
    min_samples_split: 2
  
  gradient_boosting:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 5

# Training settings
training:
  test_size: 0.2
  stratify: true  # Stratify by target
  tune: false  # Enable hyperparameter tuning

# Hyperparameter tuning (used when tune=true)
tuning:
  cv: 5  # Cross-validation folds
  scoring: accuracy
  n_jobs: -1  # Use all CPU cores
  verbose: 1
  
  # Parameter grids for tuning
  param_grids:
    random_forest:
      n_estimators: [50, 100, 200]
      max_depth: [10, 20, None]
      min_samples_split: [2, 5, 10]
      min_samples_leaf: [1, 2, 4]
    
    gradient_boosting:
      n_estimators: [50, 100, 200]
      learning_rate: [0.01, 0.1, 0.2]
      max_depth: [3, 5, 7]

# Model evaluation
evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - roc_auc
  save_confusion_matrix: true
  save_classification_report: true
```

**3. Preprocessing Configuration (`configs/preprocess_config.yaml`)**:

```yaml
# Preprocessing Configuration

# Data source
data:
  filename: WA_Fn-UseC_-Telco-Customer-Churn.csv
  target_column: Churn
  id_column: customerID

# Missing value handling
missing_values:
  strategy: drop  # Options: drop, mean, median, mode, forward_fill, backward_fill
  threshold: 0.5  # Drop columns with >50% missing values

# Feature scaling
scaling:
  method: standard  # Options: standard, minmax, robust, none
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

**4. Prediction Configuration (`configs/predict_config.yaml`)**:

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

**Install PyYAML**:

```bash
pip install PyYAML
```

**Load configuration**:

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
n_estimators = config['params']['random_forest']['n_estimators']
test_size = config['training']['test_size']

print(f"Random state: {random_state}")
print(f"N estimators: {n_estimators}")
print(f"Test size: {test_size}")
```

### Using Configuration in Code

**Example**: Training with YAML config

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
    model_params = config['params'][model_type]
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

**Best practice**: Use YAML for defaults, CLI for overrides

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

**Usage**:

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

**Example usage**:

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

## Summary of Parts 1-6

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

```output
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

## Part 7: Data Version Control with DVC

### Why Version Control Data?

**The Problem**:

```output
project/
├── data/
│   ├── train_data_v1.csv
│   ├── train_data_v2.csv
│   ├── train_data_v2_final.csv
│   ├── train_data_v2_final_ACTUALLY_FINAL.csv
│   └── train_data_v2_final_use_this_one.csv  # 😱
```

Sound familiar? Data versioning is hard:

- ❌ Large files don't belong in Git
- ❌ Manual versioning is error-prone
- ❌ Hard to track which data produced which model
- ❌ Collaboration becomes messy
- ❌ Can't easily rollback to previous versions

**The Solution**: DVC (Data Version Control)

DVC is like Git, but for data:

- ✅ Version control for large files
- ✅ Lightweight metadata in Git
- ✅ Data stored in cloud (S3, Google Drive, DagsHub)
- ✅ Track data-model relationships
- ✅ Easy collaboration
- ✅ Reproducible pipelines

**How DVC Works**:

```text
Git Repository (lightweight):
├── data/
│   └── raw.dvc             # Metadata file (small, in Git)
├── models/
│   └── model.pkl.dvc       # Metadata file (small, in Git)
└── .dvc/
    └── config              # DVC configuration (in Git)

DVC Remote Storage (cloud):
└── Large actual files:
    ├── data/raw/train.csv           # Actual data (not in Git)
    └── models/model.pkl             # Actual model (not in Git)
```

**Git tracks**: Code + DVC metadata files (`.dvc` files)  
**DVC tracks**: Actual data + models (stored in cloud)

### Part 7.1: Installing and Initializing DVC

#### Install DVC

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install DVC
pip install dvc

# Verify installation
dvc version
```

**Expected output**:

```output
DVC version: 3.63.0 (pip)
```

#### Initialize DVC in Your Project

```bash
# Navigate to project root
cd /path/to/your/project

# Initialize DVC
dvc init
```

**Expected output**:

```output
Initialized DVC repository.

You can now commit the changes to git.

+---------------------------------------------------------------------+
|                                                                     |
|        DVC has enabled anonymous aggregate usage analytics.        |
|     Read the analytics documentation (and how to opt-out) here:    |
|             <https://dvc.org/doc/user-guide/analytics>             |
|                                                                     |
+---------------------------------------------------------------------+

What's next?
------------
- Check out the documentation: <https://dvc.org/doc>
- Get help and share ideas: <https://dvc.org/chat>
- Star us on GitHub: <https://github.com/iterative/dvc>
```

#### What DVC Created

```bash
# Check what was created
ls -la .dvc/
```

**You'll see**:

```output
.dvc/
├── .gitignore          # Ignores DVC cache
├── config              # DVC configuration
└── tmp/                # Temporary files
```

```bash
# Check Git status
git status
```

**Expected**:

```output
Changes to be committed:
  new file:   .dvc/.gitignore
  new file:   .dvc/config
  new file:   .dvcignore
  modified:   .gitignore
```

DVC automatically:

- Created `.dvc/` directory for metadata
- Created `.dvc/config` for settings
- Modified `.gitignore` to ignore DVC cache
- Staged files for Git commit

#### Commit DVC Initialization

```bash
git add .dvc .dvcignore .gitignore
git commit -m "chore: Initialize DVC for data version control"
git push
```

### Part 7.2: Tracking Data with DVC

#### Understanding the Transition

Your data is currently tracked by Git:

```bash
# See what's in Git now
git ls-files data/
```

**Output**:

```output
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
data/processed/preprocessed_data.npy
data/processed/preprocessing_pipeline.pkl
data/processed/label_encoder.pkl
```

**Goal**: Move these large files to DVC tracking, keep only metadata in Git.

#### Remove Data from Git Tracking

```bash
# Remove raw data from Git (but keep files on disk)
git rm -r --cached data/raw

# Remove processed data from Git (but keep files on disk)
git rm -r --cached data/processed
```

**Note**: `--cached` flag means "remove from Git tracking but keep the actual files on your disk."

**Verify files still exist**:

```bash
ls -lh data/raw/
ls -lh data/processed/
# Files should still be there!
```

#### Add Data to DVC Tracking

```bash
# Track raw data with DVC
dvc add data/raw
```

**Expected output**:

```output
100% Adding...|████████████████████████████████|1/1 [00:XX, XX file/s]

To track the changes with git, run:

        git add data/raw.dvc data/.gitignore
```

**What happened?**

- DVC created `data/raw.dvc` (metadata file)
- DVC created/updated `data/.gitignore` (to ignore actual data)
- DVC moved actual data to `.dvc/cache/` (local cache)
- Actual files still accessible at original location (DVC creates links)

```bash
# Track processed data with DVC
dvc add data/processed
```

**Expected output**:

```output
100% Adding...|████████████████████████████████|1/1 [00:XX, XX file/s]

To track the changes with git, run:

        git add data/processed.dvc data/.gitignore
```

#### Examine DVC Metadata Files

```bash
# View the metadata file
cat data/raw.dvc
```

**Expected output**:

```yaml
outs:
- md5: 063d451250fb0faa73bc60935e759442.dir
  size: 977501
  nfiles: 1
  hash: md5
  path: raw
```

This file contains:

- **md5**: Hash of directory contents (for tracking changes)
- **size**: Total size in bytes
- **nfiles**: Number of files
- **path**: Path to data directory

```bash
# Check the .gitignore DVC created
cat data/.gitignore
```

**Expected output**:

```output
/raw
/processed
```

This tells Git to ignore the actual data directories (since DVC is now managing them).

#### Add DVC Files to Git

```bash
# Add DVC metadata files to Git
git add data/raw.dvc data/processed.dvc data/.gitignore

# Check status
git status
```

**Expected**:

```output
Changes to be committed:
  new file:   data/.gitignore
  new file:   data/processed.dvc
  new file:   data/raw.dvc
  deleted:    data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
  deleted:    data/processed/...
```

#### Commit Changes

```bash
git commit -m "feat: Track data with DVC instead of Git

- Remove large data files from Git tracking
- Add data/raw/ to DVC (977KB)
- Add data/processed/ to DVC (1.3MB)
- DVC metadata files tracked in Git

Data now version controlled with DVC, not Git."

git push
```

### Part 7.3: Setting Up Remote Storage

Your data is now tracked by DVC locally, but to collaborate or backup, you need **remote storage** (like GitHub for code, but for data).

**Remote Storage Options**:

1. **DagsHub** (Recommended) ⭐
   - Purpose-built for ML projects
   - Free tier includes DVC + MLflow hosting
   - S3-compatible (industry standard)
   - Easy setup
   - Web UI to browse data

2. **Google Drive** (Alternative)
   - Free 15GB storage
   - Familiar interface
   - Good for small projects
   - **Limitation**: OAuth issues in cloud environments (CodeSpaces)

3. **Amazon S3** (Production)
   - Industry standard
   - Highly scalable
   - Pay-as-you-go
   - Best for production

4. **Others**:
   - Google Cloud Storage
   - Azure Blob Storage
   - SSH/SFTP servers

**For this lab, we'll use DagsHub (recommended).**

---

### Part 7.4: DagsHub Setup (Recommended)

#### Why DagsHub?

- ✅ **Free for students/educators**
- ✅ **DVC + MLflow in one place** (we'll use MLflow next!)
- ✅ **S3-compatible** (same as Amazon S3 - the most common storage in industry)
- ✅ **Works in all environments** (local, cloud, CodeSpaces)
- ✅ **Web UI** to browse data versions
- ✅ **Git integration** (syncs with GitHub)

#### Step 1: Create DagsHub Account

1. Go to [https://dagshub.com/](https://dagshub.com/)
2. Click **"Sign Up"** or **"Sign in with GitHub"** (recommended - uses your GitHub account)
3. Verify your email if prompted

#### Step 2: Install DagsHub Package

```bash
# Install DagsHub CLI and authentication tools
pip install dagshub --upgrade
```

#### Step 3: Authenticate with DagsHub

```bash
# Login to DagsHub (creates authentication token)
dagshub login
```

**This will**:

1. Open a browser window (or provide a URL to open)
2. Ask you to authorize DagsHub CLI
3. Prompt you to select token expiration time

**Important Notes**:

- ⚠️ **Token Expiration**: Choose a timeframe that covers your course duration (e.g., 3 months for a semester course). The token will expire after this period.
- ⚠️ **New CodeSpaces Instances**: If you create a new CodeSpaces environment, you'll need to run `dagshub login` again to re-authenticate.
- ✅ **Token Storage**: The authentication token is stored locally in `~/.dagshub/config` (not in your project, not in Git).

**Expected output**:

```output
DagsHub login successful!
Authentication token saved to ~/.dagshub/config
```

#### Step 4: Create DagsHub Repository

1. Click **"+ New Repository"** (top right)
2. Fill in details:
   - **Repository name**: Match your GitHub repo name
   - **Description**: "Describe your problem"
   - **Visibility**: Public or Private (your choice; private if you are asked by data provider not to share)
   - **Initialize with**: Leave all unchecked (we already have a repo)
3. Click **"Create Repository"**

You'll see an empty repository page with setup instructions.

**Example (instructor's repo)**:

```url
https://dagshub.com/ajallooe/cmpt2500f25-project-tutorial
```

#### Step 5: Get DagsHub Credentials

On your DagsHub repository page:

1. Look for **"Connection credentials"** box (usually bottom right)
2. Click **"Simple Data Upload"** tab
3. You'll see:
   - **Bucket name**: Your repository name
   - **Endpoint URL**: `https://dagshub.com/api/v1/repo-buckets/s3/your-username`
   - **Access Key ID**: (a token)
   - **Secret Access Key**: (same token - DagsHub uses the same value for both)
   - **Region**: `us-east-1`

**Keep this page open - you'll need these credentials!**

#### Step 6: Configure DVC Remote

Back in your terminal:

```bash
# Add DagsHub as DVC remote
# Replace with YOUR values from DagsHub
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/your-username/your-project-name.s3

# Set as default remote
dvc remote default origin
```

**Example (with actual values)**:

```bash
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/ajallooe/cmpt2500f25-project-tutorial.s3
dvc remote default origin
```

#### Step 7: Add Credentials (Stored Locally Only)

```bash
# Add credentials (replace YOUR_TOKEN with actual token from DagsHub)
dvc remote modify origin --local access_key_id YOUR_TOKEN
dvc remote modify origin --local secret_access_key YOUR_TOKEN
```

**Important**: The `--local` flag stores credentials in `.dvc/config.local`, which is automatically ignored by Git. Your credentials stay secure on your machine only!

#### Step 8: Verify Configuration

```bash
# Check main config (will be committed to Git)
cat .dvc/config
```

**Expected output**:

```output
[core]
    remote = origin
['remote "origin"']
    url = s3://dvc
    endpointurl = https://dagshub.com/your-username/your-repo.s3
```

```bash
# Check local config (NOT committed - has credentials)
cat .dvc/config.local
```

**Expected output**:

```output
['remote "origin"']
    access_key_id = YOUR_TOKEN
    secret_access_key = YOUR_TOKEN
```

```bash
# Verify remote is set
dvc remote list
```

**Expected output**:

```output
origin  s3://dvc    (default)
```

#### Step 9: Commit Remote Configuration

```bash
# Add config to Git (NOT config.local - that's gitignored)
git add .dvc/config
git commit -m "chore: Configure DagsHub as DVC remote storage

- Add DagsHub S3-compatible remote
- Set as default remote
- Credentials stored locally (not committed)"

git push
```

#### Step 10: Push Data to DagsHub

```bash
# Push data to remote storage
dvc push
```

**Expected output**:

```output
Collecting
Pushing
2 files pushed
```

This uploads:

- `data/raw/` (955KB)
- `data/processed/` (1.3MB)

**First push might take a minute depending on your internet speed.**

#### Step 11: Verify Upload

```bash
# Check DVC status
dvc status -c
```

**Expected output**:

```output
Cache and remote 'origin' are in sync.
```

This confirms your local data and remote data match!

**Optional**: Check DagsHub Web UI

1. Go to your DagsHub repository
2. Click **"Files"** tab
3. Navigate to `data/` directory
4. You should see `.dvc` metadata files (Git tracks these)
5. The actual data is in DagsHub's storage (not visible in files, but in Storage tab)

**Note**: DagsHub UI can take a few minutes to sync. The important thing is that `dvc status -c` shows everything is in sync.

### Part 7.5: Testing DVC Workflow

Let's test that DVC actually works by simulating a fresh clone:

#### Simulate Fresh Clone

```bash
# Remove local cache
rm -rf .dvc/cache

# Verify cache is gone
ls -la .dvc/
# Should NOT see cache/ directory

# Remove actual data
rm -rf data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
rm -rf data/processed/*.pkl data/processed/*.npy

# Verify data is gone
ls -la data/raw/
ls -la data/processed/
# Should be empty
```

#### Pull Data from DagsHub

```bash
# Pull data from remote
dvc pull
```

**Expected output**:

```output
Collecting
Fetching
2 files fetched
```

#### Verify Data Restored

```bash
# Check if data is back
ls -lh data/raw/
ls -lh data/processed/
```

**Expected**: All your files should be back with correct sizes!

```output
data/raw/:
  WA_Fn-UseC_-Telco-Customer-Churn.csv (955KB)

data/processed/:
  preprocessed_data.npy (1.2MB)
  preprocessing_pipeline.pkl (48KB)
  label_encoder.pkl (484B)
```

**Success!** ✅ DVC is working correctly. You can now:

- Version control your data
- Collaborate with teammates (they just `dvc pull`)
- Rollback to previous data versions
- Track which data produced which model

---

### Part 7.6: Google Drive Setup (Alternative)

**⚠️ Important Note**: Google Drive remote has limitations in cloud environments (GitHub CodeSpaces, AWS Cloud9, etc.) due to OAuth authentication restrictions. **DagsHub is strongly recommended for cloud development.**

Google Drive works well for:

- ✅ Local development (your own computer)
- ✅ Small projects
- ✅ Personal learning

Use Google Drive only if:

- You're working on your local machine (not cloud)
- Your project is small (<15GB)
- You want to learn alternative remotes

#### Prerequisites for Google Drive

**If you want to use Google Drive, you'll need to**:

1. **Install Google Drive support**:

   ```bash
   pip install dvc-gdrive
   ```

2. **Create a Google Drive folder**:
   - Go to [https://drive.google.com](https://drive.google.com)
   - Create folder: `dvc-your-project-name`
   - Share with team members and instructor
   - Get the folder ID from URL (after `/folders/`)

3. **Configure DVC remote**:

   ```bash
   dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID
   ```

4. **Authenticate (local machine only)**:

   ```bash
   dvc push
   # This will open browser for Google OAuth
   # Follow prompts to authenticate
   ```

#### OAuth Limitation in Cloud Environments

**Why it doesn't work in CodeSpaces/cloud**:

When you run `dvc push` with Google Drive, it tries to:

1. Open a browser window for Google authentication
2. Ask you to grant permissions to PyDrive2 (DVC's Google Drive library)
3. Get an authorization code back

**The problem**:

- Cloud environments (CodeSpaces, Cloud9) can't open browsers
- Google blocks "unverified apps" for security
- PyDrive2 is considered an "unverified app"

**Error you'll see**:

>This app is blocked
>This app tried to access sensitive info in your Google Account.
>To keep your account safe, Google blocked this access.

#### Workaround: Service Account (Advanced)

For Google Drive to work in cloud environments, you need to:

1. Create a Google Cloud Project
2. Enable Google Drive API
3. Create a service account
4. Generate credentials JSON file
5. Share your Drive folder with service account email
6. Configure DVC with service account JSON

**This process**:

- Takes 15-20 minutes to set up
- Requires Google Cloud Console access
- Is more complex than DagsHub
- **Not tested in CodeSpaces for this lab**

**We provide this information for reference, but recommend DagsHub for ease of use.**

---

### Part 7.7: DVC Branch Management

Remember we created a `dvc-google-drive` branch? Let's understand the branching strategy:

#### Main Branch (DagsHub)

```bash
# Ensure you're on main
git checkout main

# Verify DagsHub remote
dvc remote list
```

**Output**:

```output
origin  s3://dvc    (default)
```

**This branch is**:

- ✅ Fully working and tested
- ✅ Recommended for all students
- ✅ Works in cloud environments
- ✅ Production-ready

#### Google Drive Branch (Alternative)

```bash
# Switch to Google Drive branch (for reference)
git checkout dvc-google-drive

# Verify Google Drive remote
dvc remote list
```

**Output**:

```output
gdrive  gdrive://FOLDER_ID    (default)
```

**This branch is**:

- ⚠️ For local development only
- ⚠️ OAuth issues in cloud environments
- ⚠️ Documented but not fully tested
- ℹ️ Learning resource for alternative remotes

**Don't worry about this branch for now. Stick with main (DagsHub)!**

---

### Part 7.8: DVC Workflow Summary

#### Daily Workflow

**1. Make changes to data**:

```bash
# Preprocess data (creates/updates files)
python -m src.preprocess --input data/raw/new_data.csv

# DVC notices files changed
dvc status
```

**2. Track changes with DVC**:

```bash
# Add changed files to DVC
dvc add data/processed

# Commit DVC metadata to Git
git add data/processed.dvc
git commit -m "feat: Update processed data with new preprocessing"
git push
```

**3. Push data to remote**:

```bash
# Upload actual data to DagsHub
dvc push
```

**4. Teammates get your changes**:

```bash
# Teammate pulls code
git pull

# Teammate pulls data
dvc pull
```

#### Common Commands

```bash
# Check what changed
dvc status

# Check if local and remote are in sync
dvc status -c

# Add/update data tracking
dvc add data/raw
dvc add data/processed

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# List remotes
dvc remote list

# Check out specific data version (like git checkout for data)
git checkout <commit-hash>
dvc pull
```

### Part 7.9: Updating requirements.txt

Don't forget to add DVC to your dependencies!

```bash
# Check if DVC is already in requirements.txt
grep dvc requirements.txt
```

If not present, add it:

```txt
# Add to requirements.txt

# Data Version Control (DagsHub remote - recommended)
dvc==3.63.0
dagshub==0.6.3
```

**Note**: If using Google Drive alternative, add:

```txt
dvc-gdrive==3.0.1
```

**Commit the update**:

```bash
git add requirements.txt
git commit -m "chore: Add DVC dependencies to requirements.txt"
git push
```

---

### Part 7.10: DVC Best Practices

✅ **DO**:

- Track large files (>10MB) with DVC
- Track data and trained models with DVC
- Keep small artifacts (<1MB) in Git (preprocessing_pipeline.pkl is borderline)
- Commit `.dvc` files to Git
- Use `dvc push` after `dvc add`
- Document remote setup in README

❌ **DON'T**:

- Track code with DVC (use Git for code)
- Commit `.dvc/cache/` to Git (it's gitignored for a reason)
- Commit `.dvc/config.local` to Git (contains credentials!)
- Forget to push data (`dvc push` after changes)
- Mix data versions across branches without care

### Part 7.11: Troubleshooting DVC

**Problem**: `dvc push` fails with "Permission denied"

**Solution**: Check credentials:

```bash
cat .dvc/config.local
# Verify tokens are correct
```

**Problem**: `dvc pull` says "Unable to find file"

**Solution**: Ensure remote has the data:

```bash
dvc status -c
# If out of sync, check Git commit and data version match
```

**Problem**: "Output 'data/raw' is already tracked by SCM"

**Solution**: Remove from Git first:

```bash
git rm -r --cached data/raw
dvc add data/raw
```

**Problem**: Large data file accidentally in Git

**Solution**: Remove from history:

```bash
# Use BFG or git filter-branch (advanced)
# Better: Prevent by using .gitignore from start
```

---

## Summary of Part 7: DVC

Congratulations! You've now:

✅ **Installed and initialized DVC**
✅ **Tracked data with DVC** (removed from Git)
✅ **Set up DagsHub remote** (S3-compatible storage)
✅ **Pushed data to cloud storage**
✅ **Tested pull workflow** (simulated fresh clone)
✅ **Understood Google Drive alternative** (with limitations)
✅ **Updated requirements.txt** with DVC dependencies

### Updated Project Structure

```output
your-project/
├── .dvc/
│   ├── .gitignore          # Ignores cache
│   ├── config              # Remote config (IN GIT)
│   ├── config.local        # Credentials (NOT in Git)
│   └── cache/              # Local data cache (NOT in Git)
├── data/
│   ├── .gitignore          # Created by DVC
│   ├── raw.dvc             # Metadata (IN GIT)
│   ├── processed.dvc       # Metadata (IN GIT)
│   ├── raw/                # Actual data (IN DVC CACHE)
│   └── processed/          # Actual data (IN DVC CACHE)
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── models/
├── requirements.txt        # Now includes: dvc, dagshub
└── README.md
```

**What's tracked where**:

| Item | Git | DVC | DagsHub |
|------|-----|-----|---------|
| Code (`.py`) | ✅ Yes | ❌ No | ✅ Synced from Git |
| Config (`.yaml`) | ✅ Yes | ❌ No | ✅ Synced from Git |
| DVC metadata (`.dvc`) | ✅ Yes | ❌ No | ✅ Synced from Git |
| Data files | ❌ No | ✅ Yes | ✅ Yes (storage) |
| Model files | ❌ No | ✅ Yes | ✅ Yes (storage) |
| Credentials | ❌ No | 🔒 Local only | ❌ No |

---

## Next: MLflow for Experiment Tracking

In the next section, we'll add MLflow to track:

- Hyperparameters
- Training metrics
- Model performance
- Experiment comparisons

Stay tuned! 🚀

---

*Lab 02 Instructions*  
*CMPT 2500: Machine Learning Deployment and Software Development*  
*NorQuest College*
