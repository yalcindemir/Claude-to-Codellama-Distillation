#!/usr/bin/env python3
"""
Simple test for imports
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

print(f"Testing imports from: {src_path}")

try:
    # Try basic imports first
    import yaml
    print("✅ yaml imported")
except ImportError as e:
    print(f"❌ yaml failed: {e}")

try:
    import pandas
    print("✅ pandas imported")
except ImportError as e:
    print(f"❌ pandas failed: {e}")

try:
    from claude_client import ClaudeConfig
    print("✅ claude_client imported")
except ImportError as e:
    print(f"❌ claude_client failed: {e}")

try:
    from dataset_generator import DatasetConfig
    print("✅ dataset_generator imported")
except ImportError as e:
    print(f"❌ dataset_generator failed: {e}")

print("\n📋 For notebook usage, add this cell at the top:")
print("```python")
print("import sys")
print("import os")
print("sys.path.append('./src')")
print("```")
