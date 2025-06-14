"""
Setup script for Claude-to-CodeLlama Knowledge Distillation project.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

setup(
    name="claude-to-codellama-distillation",
    version="1.0.0",
    author="Yalçın DEMIR",
    author_email="yalcin.demir@idias.com",
    description="Knowledge distillation system transferring Claude Opus 4's code generation capabilities to Code Llama 7B",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yalcindemir/claude-to-codellama-distillation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
        ],
        "colab": [
            "google-colab",
        ],
    },
    entry_points={
        "console_scripts": [
            "claude-distill=src.cli:main",
        ],
    },
    keywords=[
        "knowledge distillation",
        "code generation", 
        "claude",
        "code llama",
        "machine learning",
        "nlp",
        "artificial intelligence",
        "transformers",
        "large language models"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yalcindemir/Claude-to-Codellama-Distillation/issues",
        "Documentation": "https://github.com/yalcindemir/Claude-to-Codellama-Distillation/tree/main/docs",
        "Source": "https://github.com/yalcindemir/Claude-to-Codellama-Distillation",
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.md", "*.txt"],
    },
    zip_safe=False,
)