#!/usr/bin/env python3
"""
Sliding Window Predictor for Predictive Models
Designed for models that look at N frames and predict what happens AFTER those frames.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from core.base import BaseModel


class SlidingWindowPredictor(BaseModel):
    """
    Sliding window predictor for models that predict future events.
    
    Logic:
    - Model sees frames [0, 1, 2, ..., N-1] and predicts what happens after frame N-1
    - First N frames get NaN (no information available for prediction)
    - Predictions start from frame N onwards
    - No averaging between overlapping windows
    """
    
    def __init__(
        self, 
        window_size: int = 32,
        stride: int = 16,
        aggregation_method: str = "mean",  # Kept for compatibility, ignored
        target_fps: Optional[float] = None,
        fill_value=None
    ):
        """
        Initialize the predictor.
        
        Args:
            window_size: Number of frames the model looks at
            stride: Number of frames to advance between windows
            aggregation_method: Ignored (no aggregation performed)
            target_fps: Target FPS for frame sampling
            fill_value: Value to use for frames without predictions (default: np.nan)
        """
        self.window_size = window_size
        self.stride = stride
        self.aggregation_method = aggregation_method
        self.target_fps = target_fps
        self.fill_value = np.nan if fill_value is None else fill_value
        
        if self.stride > self.window_size:
            raise ValueError(f"Stride ({stride}) cannot be larger than window_size ({window_size})")
    
    def create_windows(self, total_frames: int) -> List[Tuple[int, int]]:
        """Create sliding window frame ranges"""
        if total_frames <= 0:
            raise ValueError(f"Invalid total_frames: {total_frames}")
        
        windows = []
        
        if total_frames <= self.window_size:
            # Video shorter than window size
            windows.append((0, total_frames))
        else:
            # Create overlapping windows
            start = 0
            while start <= total_frames - self.window_size:
                end = start + self.window_size
                windows.append((start, end))
                start += self.stride
        
        return windows
    
    def pad_window_frames(self, frames: np.ndarray, target_size: int) -> np.ndarray:
        """Pad frames to target size if needed"""
        num_frames = frames.shape[0]
        
        if num_frames == target_size:
            return frames
        elif num_frames > target_size:
            # Take last frames if too many
            return frames[-target_size:]
        else:
            # Pad with black frames at beginning
            frame_shape = frames.shape[1:]
            padding_needed = target_size - num_frames
            black_frames = np.zeros((padding_needed, *frame_shape), dtype=frames.dtype)
            return np.concatenate([black_frames, frames], axis=0)
    
    def predict_sliding_windows(self, 
                              video_path: str, 
                              model_predict_fn,
                              preprocess_fn,
                              return_per_frame: bool = True) -> Dict[str, np.ndarray]:
        """
        Run sliding window prediction.
        
        Args:
            video_path: Path to video file
            model_predict_fn: Function that takes processed frames and returns prediction
            preprocess_fn: Function that preprocesses frame array for model
            return_per_frame: Whether to return per-frame predictions
            
        Returns:
            Dictionary with 'windows', 'per_video', and optionally 'per_frame' predictions
        """
        # Load video frames
        try:
            from .video import load_full_video_frames
            all_frames = load_full_video_frames(
                video_path=video_path,
                target_size=(224, 224),
                target_fps=self.target_fps
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load video frames from {video_path}: {e}")
        
        total_frames = len(all_frames)
        
        # Create windows
        windows = self.create_windows(total_frames)
        
        # Store results
        window_predictions = []
        prediction_targets = []  # Frame indices that predictions target
        
        # Process each window
        for i, (start_idx, end_idx) in enumerate(windows):
            try:
                # Extract frames for this window
                window_frames = all_frames[start_idx:end_idx]
                
                # Pad if necessary
                padded_frames = self.pad_window_frames(window_frames, self.window_size)
                
                # Preprocess
                processed_frames = preprocess_fn(padded_frames)
                
                # Get prediction
                prediction = model_predict_fn(processed_frames)
                
                # This prediction is for the frame that comes AFTER the window
                target_frame = end_idx
                
                window_predictions.append(prediction)
                prediction_targets.append(target_frame)
                
            except Exception as e:
                raise RuntimeError(f"Failed to process window {i} ({start_idx}-{end_idx}): {e}")
        
        # Prepare results
        results = {
            'windows': window_predictions,
            'per_video': np.mean(window_predictions) if window_predictions else 0.0
        }
        
        # Create per-frame predictions if requested
        if return_per_frame:
            frame_predictions = self._create_predictive_frame_array(
                prediction_targets, window_predictions, total_frames
            )
            results['per_frame'] = frame_predictions
        
        return results
    
    def _create_predictive_frame_array(self, 
                                     target_frames: List[int], 
                                     predictions: List[float], 
                                     total_frames: int) -> np.ndarray:
        """
        Create frame-by-frame prediction array for predictive model.
        
        Logic:
        - First window_size frames: NaN (no predictions available)
        - Frames with direct predictions: use those values
        - Frames between predictions: linear interpolation
        - Frames after last prediction: use last prediction value
        
        Args:
            target_frames: Frame indices that predictions target
            predictions: Prediction values
            total_frames: Total number of frames in video
            
        Returns:
            Array of predictions for each frame
        """
        frame_predictions = np.full(total_frames, self.fill_value, dtype=np.float32)
        
        if not target_frames:
            return frame_predictions
        
        # Assign direct predictions
        for target_frame, prediction in zip(target_frames, predictions):
            if 0 <= target_frame < total_frames:
                frame_predictions[target_frame] = prediction
        
        # Fill in values through interpolation and extension
        for i in range(total_frames):
            if np.isnan(frame_predictions[i]):
                # Find surrounding predictions
                before_targets = [t for t in target_frames if t < i]
                after_targets = [t for t in target_frames if t > i]
                
                if before_targets and after_targets:
                    # Interpolate between nearest predictions
                    before_target = max(before_targets)
                    after_target = min(after_targets)
                    
                    before_pred = predictions[target_frames.index(before_target)]
                    after_pred = predictions[target_frames.index(after_target)]
                    
                    # Linear interpolation
                    weight = (i - before_target) / (after_target - before_target)
                    frame_predictions[i] = before_pred + weight * (after_pred - before_pred)
                    
                elif before_targets:
                    # Extend last prediction forward
                    last_target = max(before_targets)
                    frame_predictions[i] = predictions[target_frames.index(last_target)]
                    
                # Note: we don't backfill from future predictions to preserve NaN region
        
        return frame_predictions
    
    def load(self) -> None:
        """Required by BaseModel - no loading needed"""
        pass
    
    def predict(self, video_path: str) -> np.ndarray:
        """Required by BaseModel - not used directly"""
        raise NotImplementedError("Use predict_sliding_windows instead")
    
    def predict_batch(self, video_paths: List[str]) -> List[np.ndarray]:
        """Required by BaseModel - not used directly"""
        raise NotImplementedError("Use predict_sliding_windows instead")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information"""
        return {
            "name": "Predictive Sliding Window",
            "window_size": self.window_size,
            "stride": self.stride,
            "target_fps": self.target_fps,
            "prediction_mode": "predictive_no_averaging",
            "first_n_frames": "NaN",
            "version": "4.0"
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration"""
        return {
            'window_size': self.window_size,
            'stride': self.stride,
            'aggregation_method': 'none',
            'target_fps': self.target_fps,
            'overlap': self.window_size - self.stride,
            'prediction_mode': 'predictive_future_events'
        }