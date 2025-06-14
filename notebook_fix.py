#!/usr/bin/env python3
"""
Fix notebook imports by updating the notebook content
"""

import json
import os

def fix_notebook():
    notebook_path = "notebooks/Claude_Code_Model_Colab.ipynb"
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the cell with imports and fix it
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source_lines = cell['source']
            source_text = ''.join(source_lines)
            
            # Fix the import cell
            if 'from dataset_generator import' in source_text:
                print("Found import cell, fixing...")
                
                new_source = [
                    "# Fix imports by adding src to path\n",
                    "import sys\n",
                    "import os\n",
                    "sys.path.append('./src')\n",
                    "sys.path.append(os.path.join(os.getcwd(), 'src'))\n",
                    "\n",
                    "import asyncio\n",
                    "from dataset_generator import DatasetGenerator, DatasetConfig\n",
                    "from claude_client import ClaudeConfig\n",
                    "\n",
                    "# Configure dataset generation\n",
                    "claude_config = ClaudeConfig(\n",
                    "    api_key=os.getenv('ANTHROPIC_API_KEY'),\n",
                    "    model='claude-3-opus-20240229',\n",
                    "    max_tokens=1024,\n",
                    "    temperature=0.1,\n",
                    "    rate_limit_rpm=30  # Conservative for Colab\n",
                    ")\n",
                    "\n",
                    "dataset_config = DatasetConfig(\n",
                    "    target_size=COLAB_CONFIG['target_size'],\n",
                    "    languages=['python', 'javascript'],  # Start with 2 languages\n",
                    "    output_dir='./data/generated'\n",
                    ")\n",
                    "\n",
                    "print(\"🏗️ Starting dataset generation...\")\n",
                    "print(f\"Target size: {dataset_config.target_size} examples\")\n",
                    "print(f\"Languages: {dataset_config.languages}\")"
                ]
                
                cell['source'] = new_source
                break
    
    # Also add a setup cell right after requirements installation
    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "id": "python_path_setup"
        },
        "outputs": [],
        "source": [
            "# Add project src to Python path\n",
            "import sys\n",
            "import os\n",
            "\n",
            "# Add src directory to path\n",
            "src_path = os.path.join(os.getcwd(), 'src')\n",
            "if src_path not in sys.path:\n",
            "    sys.path.insert(0, src_path)\n",
            "    print(f\"✅ Added {src_path} to Python path\")\n",
            "\n",
            "# Test imports\n",
            "try:\n",
            "    from claude_client import ClaudeConfig\n",
            "    print(\"✅ claude_client import successful\")\n",
            "except ImportError as e:\n",
            "    print(f\"❌ claude_client import failed: {e}\")\n",
            "\n",
            "try:\n",
            "    from dataset_generator import DatasetGenerator\n",
            "    print(\"✅ dataset_generator import successful\")\n",
            "except ImportError as e:\n",
            "    print(f\"❌ dataset_generator import failed: {e}\")"
        ]
    }
    
    # Insert the setup cell after requirements installation
    requirements_index = -1
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'source' in cell:
            source_text = ''.join(cell['source'])
            if 'pip install -r requirements.txt' in source_text:
                requirements_index = i
                break
    
    if requirements_index >= 0:
        notebook['cells'].insert(requirements_index + 1, setup_cell)
        print("Added Python path setup cell")
    
    # Write the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ Updated {notebook_path}")

def create_simple_test():
    """Create a simple test script for imports"""
    test_content = '''#!/usr/bin/env python3
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

print("\\n📋 For notebook usage, add this cell at the top:")
print("```python")
print("import sys")
print("import os")
print("sys.path.append('./src')")
print("```")
'''
    
    with open('test_imports.py', 'w') as f:
        f.write(test_content)
    
    print("✅ Created test_imports.py")

if __name__ == "__main__":
    fix_notebook()
    create_simple_test()