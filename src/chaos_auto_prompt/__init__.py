"""
chaos-auto-prompt: A framework for automated prompt optimization.

This package provides tools for optimizing LLM prompts using various strategies,
including meta-prompting, evaluation-based optimization, and budget management.
"""

__version__ = "0.1.0"

from .optimizers import MetaPrompt, PromptLearningOptimizer
from .config.settings import settings, get_settings

__all__ = [
    "MetaPrompt",
    "PromptLearningOptimizer",
    "settings",
    "get_settings",
]
