#!/usr/bin/env python3
"""Quick experiment runner for continual learning systems."""

import os
import sys
import subprocess
from pathlib import Path

def run_experiment(config_name: str, experiment_name: str = None):
    """Run a continual learning experiment."""
    config_path = f"configs/{config_name}.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    cmd = ["python", "src/main.py", "--config", config_path]
    
    if experiment_name:
        cmd.extend(["--experiment_name", experiment_name])
    
    print(f"🚀 Running experiment: {config_name}")
    print(f"📝 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Experiment completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main function."""
    print("🧠 Continual Learning Systems - Quick Experiment Runner")
    print("=" * 60)
    
    # Available experiments
    experiments = [
        ("default", "EWC Baseline"),
        ("mas_experiment", "Memory Aware Synapses"),
        ("replay_experiment", "Experience Replay")
    ]
    
    print("Available experiments:")
    for i, (config, description) in enumerate(experiments, 1):
        print(f"  {i}. {description} ({config})")
    
    print("\nOptions:")
    print("  - Run all experiments: python scripts/run_experiments.py --all")
    print("  - Run specific experiment: python scripts/run_experiments.py --config <name>")
    print("  - Interactive mode: python scripts/run_experiments.py")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            print("\n🔄 Running all experiments...")
            success_count = 0
            for config, description in experiments:
                print(f"\n{'='*40}")
                print(f"Running: {description}")
                if run_experiment(config, f"quick_{config}"):
                    success_count += 1
            print(f"\n✅ Completed {success_count}/{len(experiments)} experiments successfully!")
            
        elif sys.argv[1] == "--config" and len(sys.argv) > 2:
            config_name = sys.argv[2]
            run_experiment(config_name, f"quick_{config_name}")
            
        else:
            print("❌ Invalid arguments. Use --help for usage information.")
    else:
        # Interactive mode
        print("\n🎮 Interactive Mode")
        print("Enter experiment number (1-3) or 'q' to quit:")
        
        while True:
            try:
                choice = input("> ").strip()
                
                if choice.lower() == 'q':
                    print("👋 Goodbye!")
                    break
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(experiments):
                    config, description = experiments[choice_num - 1]
                    print(f"\n🚀 Running: {description}")
                    run_experiment(config, f"interactive_{config}")
                    print("\nEnter another experiment number or 'q' to quit:")
                else:
                    print(f"❌ Please enter a number between 1 and {len(experiments)}")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

if __name__ == "__main__":
    main()
