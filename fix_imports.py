#!/usr/bin/env python3
"""
Script to test and fix import issues for the project
"""

import sys
import os
import subprocess

def main():
    print("🔧 Fixing import issues...")
    
    # Add current directory to Python path
    current_dir = os.getcwd()
    src_dir = os.path.join(current_dir, 'src')
    
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        print(f"✅ Added {src_dir} to Python path")
    
    # Test imports
    try:
        from claude_client import ClaudeConfig, ClaudeAPIClient
        print("✅ claude_client imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import claude_client: {e}")
    
    try:
        from dataset_generator import DatasetGenerator, DatasetConfig
        print("✅ dataset_generator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import dataset_generator: {e}")
    
    try:
        from distillation_trainer import KnowledgeDistillationSystem, DistillationConfig
        print("✅ distillation_trainer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import distillation_trainer: {e}")
    
    try:
        from evaluation_system import ModelComparator, EvaluationConfig
        print("✅ evaluation_system imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import evaluation_system: {e}")
    
    # Install package in editable mode
    print("\n📦 Installing package in editable mode...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Package installed successfully")
        else:
            print(f"❌ Package installation failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Installation error: {e}")
    
    print("\n🔄 Testing imports after installation...")
    
    # Test imports again after installation
    try:
        import claude_to_codellama_distillation
        from claude_to_codellama_distillation.claude_client import ClaudeConfig
        print("✅ Package imports working correctly")
    except ImportError as e:
        print(f"❌ Package import still failing: {e}")
    
    print("\n📋 Quick fix for Jupyter/Colab:")
    print("Add this to the top of your notebook cell:")
    print("```python")
    print("import sys")
    print("import os")
    print("sys.path.append('./src')")
    print("# or")
    print("sys.path.append(os.path.join(os.getcwd(), 'src'))")
    print("```")

if __name__ == "__main__":
    main()