#!/usr/bin/env python3
"""
Core Abstract Base Classes for Modular Evaluation Framework
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import random


@dataclass
class DatasetHierarchy:
    """Hierarchical dataset categorization"""
    category: str  # e.g., "Nexar", "DAD", "Custom"
    subcategory: Optional[str] = None  # e.g., "Group_0", "Private", "Training"
    subsubcategory: Optional[str] = None  # e.g., additional nested levels
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "category": self.category,
            "subcategory": self.subcategory,
            "subsubcategory": self.subsubcategory
        }
    
    def get_full_path(self, separator: str = " > ") -> str:
        """Get full hierarchical path as string"""
        parts = [self.category]
        if self.subcategory:
            parts.append(self.subcategory)
        if self.subsubcategory:
            parts.append(self.subsubcategory)
        return separator.join(parts)
    
    def get_display_name(self) -> str:
        """Get display name for DataFrames and visualizations"""
        if self.subcategory:
            if self.subsubcategory:
                return f"{self.category}_{self.subcategory}_{self.subsubcategory}"
            return f"{self.category}_{self.subcategory}"
        return self.category


@dataclass
class Sample:
    """Single evaluation sample"""
    video_path: str
    label: int  # 0=normal, 1=accident
    sample_id: str
    metadata: Dict[str, Any]


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    @abstractmethod
    def load(self) -> None:
        """Initialize model (from checkpoint/pretrained)"""
        pass
    
    @abstractmethod
    def predict(self, video_path: str) -> np.ndarray:
        """Return frame-level predictions for a single video
        
        Args:
            video_path: Path to video file
            
        Returns:
            np.ndarray: Frame-level predictions, shape (num_frames,)
        """
        pass
    
    @abstractmethod
    def predict_batch(self, video_paths: List[str]) -> List[np.ndarray]:
        """Return frame-level predictions for multiple videos
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List[np.ndarray]: List of frame-level predictions
        """
        pass
    
    @abstractmethod 
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata
        
        Returns:
            Dict containing model name, version, etc.
        """
        pass


class BaseDataset(ABC):
    """Abstract base class for all datasets"""
    
    @abstractmethod
    def load_samples(self) -> List[Sample]:
        """Load dataset samples
        
        The dataset should internally determine which samples to load
        based on its configuration (split, group, usage, etc.)
        
        Returns:
            List[Sample]: List of samples for this dataset instance
        """
        pass
    
    @abstractmethod
    def get_sample_by_id(self, sample_id: str) -> Sample:
        """Get individual sample by ID
        
        Args:
            sample_id: Unique sample identifier
            
        Returns:
            Sample: The requested sample
        """
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset metadata
        
        Returns:
            Dict containing dataset name, size, classes, etc.
        """
        pass
    
    def get_hierarchy(self) -> DatasetHierarchy:
        """Return dataset hierarchy information
        
        Returns:
            DatasetHierarchy: Hierarchical categorization of this dataset
        """
        # Default implementation - subclasses should override for specific hierarchies
        info = self.get_dataset_info()
        return DatasetHierarchy(category=info.get("name", "Unknown"))
    
    def get_hierarchical_name(self) -> str:
        """Get hierarchical display name for this dataset
        
        Returns:
            str: Display name incorporating hierarchy
        """
        return self.get_hierarchy().get_display_name()
    
    def _apply_max_samples(self, samples: List[Sample], max_samples: Optional[int], 
                          random_seed: int = 42) -> List[Sample]:
        """Apply max_samples random sampling to a list of samples
        
        Args:
            samples: List of samples to sample from
            max_samples: Maximum number of samples to return (None for no limit)
            random_seed: Random seed for reproducible sampling
            
        Returns:
            List[Sample]: Randomly sampled subset of samples (or all if max_samples is None/larger)
        """
        if max_samples is None or len(samples) <= max_samples:
            return samples
        
        # Set random seed for reproducible sampling
        random.seed(random_seed)
        return random.sample(samples, max_samples)


class BaseMetric(ABC):
    """Abstract base class for all metrics"""
    
    @property
    @abstractmethod
    def requires_frame_level(self) -> bool:
        """Whether this metric requires frame-level predictions"""
        pass
    
    @abstractmethod
    def compute(self, predictions: np.ndarray, labels: np.ndarray, 
               metadata: Optional[List[Dict[str, Any]]] = None) -> float:
        """Compute metric value
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            metadata: Optional list of metadata dicts, one per sample.
                     Each dict can contain sample-specific information like:
                     - fps: Frame rate for temporal metrics
                     - accident_frame: Frame index of accident
                     - video_id: Video identifier
                     - group: Dataset group/category
                     - Any other sample-specific metadata
            
        Returns:
            float: Computed metric value
        """
        pass
    
    @abstractmethod
    def get_metric_info(self) -> Dict[str, Any]:
        """Return metric metadata
        
        Returns:
            Dict containing metric name, range, interpretation, etc.
        """
        pass