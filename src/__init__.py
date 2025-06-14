"""
Claude-to-CodeLlama Bilgi Damıtımı Projesi

Claude Opus 4'ün kod üretim yeteneklerini bilgi damıtımı yoluyla
Code Llama 7B'ye aktaran kapsamlı bir sistem.
"""

__version__ = "1.0.0"
__author__ = "Yalçın DEMIR"
__description__ = "Claude-to-CodeLlama Bilgi Damıtma (Knowledge Distillation)"

from .claude_client import ClaudeAPIClient, ClaudeConfig
from .dataset_generator import DatasetGenerator, DatasetConfig
from .distillation_trainer import KnowledgeDistillationSystem, DistillationConfig
from .evaluation_system import ModelComparator, EvaluationConfig

__all__ = [
    "ClaudeAPIClient",
    "ClaudeConfig", 
    "DatasetGenerator",
    "DatasetConfig",
    "KnowledgeDistillationSystem",
    "DistillationConfig",
    "ModelComparator",
    "EvaluationConfig"
]