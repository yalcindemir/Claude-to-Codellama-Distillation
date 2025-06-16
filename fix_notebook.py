#!/usr/bin/env python3
"""
Fix notebook JSON schema issues
"""

import json
import os

def fix_notebook():
    notebook_path = "notebooks/Claude_Code_Model_Colab.ipynb"
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print(f"✅ Loaded notebook with {len(notebook['cells'])} cells")
    
    fixed_count = 0
    
    # Fix each cell
    for i, cell in enumerate(notebook['cells']):
        # Remove 'outputs' field from markdown cells (it's invalid)
        if cell['cell_type'] == 'markdown' and 'outputs' in cell:
            del cell['outputs']
            print(f"✅ Fixed cell {i}: removed outputs from markdown")
            fixed_count += 1
        
        # Ensure execution_count is null for markdown cells
        if cell['cell_type'] == 'markdown' and 'execution_count' in cell:
            del cell['execution_count']
            print(f"✅ Fixed cell {i}: removed execution_count from markdown")
            fixed_count += 1
        
        # Ensure code cells have proper structure
        if cell['cell_type'] == 'code':
            if 'execution_count' not in cell:
                cell['execution_count'] = None
            if 'outputs' not in cell:
                cell['outputs'] = []
    
    # Fix the first cell (Colab badge) to be properly structured
    if len(notebook['cells']) > 0:
        first_cell = notebook['cells'][0]
        if first_cell['cell_type'] == 'markdown':
            # Ensure proper structure
            notebook['cells'][0] = {
                'cell_type': 'markdown',
                'metadata': {
                    'id': 'view-in-github',
                    'colab_type': 'text'
                },
                'source': [
                    '<a href="https://colab.research.google.com/github/yalcindemir/Claude-to-Codellama-Distillation/blob/main/notebooks/Claude_Code_Model_Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
                ]
            }
            print("✅ Fixed Colab badge cell")
            fixed_count += 1
    
    # Write the fixed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ Fixed {fixed_count} issues in notebook")
    print(f"✅ Notebook saved: {notebook_path}")

if __name__ == "__main__":
    fix_notebook()