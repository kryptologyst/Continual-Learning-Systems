#!/usr/bin/env python3
"""Quick start script for continual learning systems."""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main function for quick start."""
    print("🧠 Continual Learning Systems - Quick Start")
    print("=" * 50)
    print("Author: kryptologyst")
    print("GitHub: https://github.com/kryptologyst")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("src/main.py"):
        print("❌ Please run this script from the project root directory")
        print("   (where src/main.py is located)")
        sys.exit(1)
    
    print("🚀 Available options:")
    print()
    print("1. Run baseline experiment (EWC)")
    print("   python src/main.py --config configs/default.yaml")
    print()
    print("2. Run Memory Aware Synapses experiment")
    print("   python src/main.py --config configs/mas_experiment.yaml")
    print()
    print("3. Run Experience Replay experiment")
    print("   python src/main.py --config configs/replay_experiment.yaml")
    print()
    print("4. Launch interactive demo")
    print("   streamlit run demo/app.py")
    print()
    print("5. Run all experiments")
    print("   python scripts/run_experiments.py --all")
    print()
    print("6. Run tests")
    print("   pytest tests/")
    print()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install torch")
        return
    
    try:
        import numpy
        print(f"✅ NumPy {numpy.__version__}")
    except ImportError:
        print("❌ NumPy not installed. Run: pip install numpy")
        return
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not installed. Run: pip install scikit-learn")
        return
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib not installed. Run: pip install matplotlib")
        return
    
    print()
    print("✅ All core dependencies are available!")
    print()
    print("🎯 Quick test - running a small experiment...")
    
    # Run a quick test
    try:
        result = subprocess.run([
            "python", "src/main.py", 
            "--config", "configs/default.yaml",
            "--experiment_name", "quick_test"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Quick test completed successfully!")
            print("📊 Check the results/ directory for outputs")
        else:
            print("❌ Quick test failed:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Quick test timed out (this is normal for longer experiments)")
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
    
    print()
    print("🎉 Setup complete! You're ready to explore continual learning!")
    print()
    print("📚 Next steps:")
    print("   - Read the README.md for detailed documentation")
    print("   - Try the interactive demo: streamlit run demo/app.py")
    print("   - Run experiments: python scripts/run_experiments.py")
    print("   - Explore the code in src/ directory")

if __name__ == "__main__":
    main()
