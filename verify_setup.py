"""
Setup Verification Script
Run this to check if your environment is properly configured for optimization
"""

import sys
from pathlib import Path

print("="*70)
print("BBB Prediction Optimization - Setup Verification")
print("="*70)
print()

# Check Python version
print(f"Python version: {sys.version}")
python_version = sys.version_info
if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
    print("❌ ERROR: Python 3.8+ required")
    sys.exit(1)
else:
    print("✓ Python version OK")
print()

# Check required packages
print("Checking required packages...")
required_packages = {
    'torch': 'PyTorch',
    'torch_geometric': 'PyTorch Geometric',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'sklearn': 'scikit-learn',
    'optuna': 'Optuna',
    'cma': 'CMA-ES'
}

missing_packages = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"✓ {name}")
    except ImportError:
        print(f"❌ {name} not found")
        missing_packages.append(package)

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print("\nInstall with:")
    print("pip install torch torch-geometric pandas numpy scikit-learn optuna cma")
    sys.exit(1)

print("\n✓ All required packages installed")
print()

# Check directory structure
print("Checking directory structure...")
required_dirs = [
    'configs',
    'data',
    'graph',
    'model',
    'optimizations',
    'scripts'
]

missing_dirs = []
for dir_name in required_dirs:
    if not Path(dir_name).exists():
        print(f"❌ Directory '{dir_name}/' not found")
        missing_dirs.append(dir_name)
    else:
        print(f"✓ {dir_name}/")

if missing_dirs:
    print(f"\n❌ Missing directories: {', '.join(missing_dirs)}")
    print("\nCreate them or check you're in the project root directory")
    sys.exit(1)

print("\n✓ All required directories present")
print()

# Check for key files
print("Checking key files...")
required_files = [
    'configs/base_config.py',
    'configs/graph_configs.py',
    'configs/predictor_config.py',
    'graph/featurizer.py',
    'model/gat.py',
    'model/gcn.py',
    'model/predictor.py',
    'scripts/train.py',
    'scripts/evaluate.py',
    'optimizations/bayesian_optimization.py'
]

missing_files = []
for file_path in required_files:
    if not Path(file_path).exists():
        print(f"❌ {file_path}")
        missing_files.append(file_path)
    else:
        print(f"✓ {file_path}")

if missing_files:
    print(f"\n❌ Missing files: {', '.join(missing_files)}")
    sys.exit(1)

print("\n✓ All key files present")
print()

# Check data files
print("Checking data files...")
try:
    from configs.base_config import BaseSettings
    
    data_files = [
        ('Training data', BaseSettings.TRAIN_DATA),
        ('Validation data', BaseSettings.VAL_DATA),
        ('Test data', BaseSettings.TEST_DATA)
    ]
    
    missing_data = []
    for name, path in data_files:
        if not path.exists():
            print(f"❌ {name}: {path}")
            missing_data.append(str(path))
        else:
            print(f"✓ {name}: {path}")
    
    if missing_data:
        print(f"\n❌ Missing data files")
        print("Update paths in configs/base_config.py or place files in data/ directory")
        sys.exit(1)
    
    print("\n✓ All data files found")
    print()
    
except Exception as e:
    print(f"❌ Error checking data files: {e}")
    sys.exit(1)

# Try to import main modules
print("Testing module imports...")
try:
    from configs.base_config import BaseSettings
    print("✓ configs.base_config")
    
    from graph.featurizer import MoleculeDataset
    print("✓ graph.featurizer")
    
    from model.gat import GAT
    print("✓ model.gat")
    
    from scripts.train import train_model
    print("✓ scripts.train")
    
    from scripts.evaluate import test_model
    print("✓ scripts.evaluate")
    
    print("\n✓ All modules import successfully")
    print()
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure you're running from the project root directory!")
    sys.exit(1)

# Test data loading
print("Testing data loading...")
try:
    import pandas as pd
    from configs.base_config import BaseSettings
    
    train_df = pd.read_csv(BaseSettings.TRAIN_DATA)
    print(f"✓ Loaded {len(train_df)} training samples")
    
    # Check required columns
    if 'SMILES' not in train_df.columns:
        print("❌ 'SMILES' column not found in training data")
        sys.exit(1)
    
    if 'target' not in train_df.columns:
        print("❌ 'target' column not found in training data")
        sys.exit(1)
    
    print("✓ Data has required columns (SMILES, target)")
    print()
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

# Final summary
print("="*70)
print("✓ VERIFICATION COMPLETE - SETUP IS CORRECT!")
print("="*70)
print()
print("You can now run optimizations:")
print("  python optimizations/bayesian_optimization.py --model GAT --n_trials 10")
print("  python optimizations/hyperband_optimization.py --model GAT --n_trials 10")
print("  python optimizations/cmaes_optimization.py --model GAT --n_iterations 5")
print()
print("Start with small trial counts to test, then increase for full runs.")
print("="*70)
