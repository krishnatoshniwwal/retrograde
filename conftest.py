"""
conftest.py — pytest configuration for the retrograde project.

Adds the project root to sys.path so `from src.xxx import ...` works
without installing the package.
"""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))
