"""
BADAS: V-JEPA2 Based Advanced Driver Assistance System

A state-of-the-art collision prediction model for ego-centric threat detection
in real-world driving scenarios.
"""

__version__ = "1.0.0"
__author__ = "Nexar AI Research"
__license__ = "Apache 2.0"

from .badas_loader import load_badas_model, BADASModel, preprocess_video

__all__ = [
    "load_badas_model",
    "BADASModel", 
    "preprocess_video"
]