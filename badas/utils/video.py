#!/usr/bin/env python3
"""
Shared video processing utilities extracted from video_inference.py
"""

import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Optional, Dict, Any
import re

# Import training components - reuse existing logic
try:
    from train.video_training import detect_model_type, EnhancedVideoClassifier
    HAS_TRAINING_MODULES = True
except ImportError:
    HAS_TRAINING_MODULES = False

# Try to import processors
try:
    from transformers import VideoMAEImageProcessor
    HAS_VIDEOMAE_PROCESSOR = True
except ImportError:
    HAS_VIDEOMAE_PROCESSOR = False

try:
    from transformers import AutoImageProcessor, AutoVideoProcessor
    HAS_AUTO_PROCESSOR = True
except ImportError:
    HAS_AUTO_PROCESSOR = False


def get_device() -> torch.device:
    """Get the appropriate device for model loading"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_model_file(model_path: str) -> None:
    """Validate that model file exists and is readable"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.isfile(model_path):
        raise ValueError(f"Path is not a file: {model_path}")
    
    # Try to load checkpoint to validate format
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model' not in checkpoint and 'model_state_dict' not in checkpoint:
            raise ValueError(f"Invalid checkpoint format - missing model weights: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {model_path} - {e}")


def get_processor_for_model(model_name: str):
    """Get appropriate processor for model type - extracted from video_inference.py"""
    if not HAS_TRAINING_MODULES:
        raise RuntimeError("Training modules not available - cannot detect model type")
    
    model_info = detect_model_type(model_name)
    
    if model_info['processor_type'] == 'auto_video' and HAS_AUTO_PROCESSOR:
        try:
            processor = AutoVideoProcessor.from_pretrained(model_name)
            print("Using AutoVideoProcessor for V-JEPA 2")
            return processor
        except Exception as e:
            print(f"Failed to load AutoVideoProcessor ({e})")
    
    elif model_info['processor_type'] == 'videomae' and HAS_VIDEOMAE_PROCESSOR:
        try:
            processor = VideoMAEImageProcessor.from_pretrained(model_name)
            print("Using VideoMAEImageProcessor")
            return processor
        except Exception as e:
            print(f"Failed to load VideoMAEImageProcessor ({e})")
    
    elif model_info['processor_type'] == 'auto_image' and HAS_AUTO_PROCESSOR:
        try:
            processor = AutoImageProcessor.from_pretrained(model_name)
            print("Using AutoImageProcessor")
            return processor
        except Exception as e:
            print(f"Failed to load AutoImageProcessor ({e})")
    
    print("Using manual transforms")
    return None


def get_transform_for_model(model_name: str, img_size: int = 224) -> A.Compose:
    """Get transformation pipeline for model type"""
    if not HAS_TRAINING_MODULES:
        # Fallback transform
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    model_info = detect_model_type(model_name)
    
    # Use V-JEPA crop size if available
    if model_info['is_vjepa2'] and model_info['crop_size']:
        img_size = model_info['crop_size']
    
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_model_args(model_config: Dict[str, Any]) -> Any:
    """Create argparse.Namespace-like object for model configuration"""
    class ModelArgs:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
            
            # Set default values that might be missing
            if not hasattr(self, 'num_classes'):
                self.num_classes = 2
            if not hasattr(self, 'vjepa2_crop_size'):
                self.vjepa2_crop_size = 256  
            if not hasattr(self, 'feature_combination_method'):
                self.feature_combination_method = 'concat'
            if not hasattr(self, 'future_prediction_seconds'):
                self.future_prediction_seconds = 1.0
            if not hasattr(self, 'original_fps'):
                self.original_fps = 4
            if not hasattr(self, 'temporal_hidden_dim'):
                self.temporal_hidden_dim = 512
            if not hasattr(self, 'temporal_num_heads'):
                self.temporal_num_heads = 8
            if not hasattr(self, 'head_hidden_dim'):
                self.head_hidden_dim = 512
            if not hasattr(self, 'head_num_layers'):
                self.head_num_layers = 2
            if not hasattr(self, 'head_num_heads'):
                self.head_num_heads = 8
            if not hasattr(self, 'head_dropout'):
                self.head_dropout = 0.1
    
    return ModelArgs(model_config)


def load_vjepa_model(model_name: str, checkpoint_path: Optional[str] = None, device: Optional[torch.device] = None) -> torch.nn.Module:
    """Load V-JEPA model using project's loading logic - extracted from video_inference.py"""
    if not HAS_TRAINING_MODULES:
        raise RuntimeError("Training modules not available - cannot load V-JEPA model")
    
    if device is None:
        device = get_device()
    
    # Validate checkpoint if provided
    if checkpoint_path:
        validate_model_file(checkpoint_path)
    
    try:
        # Default model configuration
        model_config = {
            "model_name": model_name,
            "frame_count": 32,
            "img_size": 224,
            "use_frame_level": False,
            "use_future_prediction": False,
            "use_classification_model": False,
            "head_type": "linear", 
            "temporal_method": "mean",
            "num_classes": 2
        }
        
        if checkpoint_path:
            # Load checkpoint first to get saved config
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if 'config' in checkpoint:
                saved_config = checkpoint['config']
                #print(f"✅ Found saved config in checkpoint")
                
                # Override model_config with saved parameters for architecture
                architecture_params = [
                    'head_type', 'head_hidden_dim', 'head_num_layers', 'head_num_heads',
                    'head_dropout', 'temporal_method', 'temporal_hidden_dim', 'temporal_num_heads',
                    'use_frame_level', 'use_future_prediction', 'use_classification_model',
                    'feature_combination_method', 'future_prediction_seconds', 'original_fps',
                    'frame_count', 'img_size', 'vjepa2_crop_size', 'num_classes'
                ]
                
                for param in architecture_params:
                    if param in saved_config:
                        model_config[param] = saved_config[param]
                        #print(f"   📝 Loaded {param}: {saved_config[param]}")
            else:
                print("⚠️ No saved config found in checkpoint, using default parameters")
        
        # Create model arguments
        model_args = create_model_args(model_config)
        model_info = detect_model_type(model_name)
        
        # Create model using the enhanced classifier
        model = EnhancedVideoClassifier(model_args, model_info).to(device)
        
        if checkpoint_path:
            # Load checkpoint weights
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise KeyError("Checkpoint missing model weights")
        
        model.eval()
        return model
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cannot find model file: {e}")
    except KeyError as e:
        raise ValueError(f"Missing key in checkpoint: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load V-JEPA model: {e}")

def preprocess_video_frames(video_path: str, target_frames: int = 32, 
                                 target_size: Tuple[int, int] = (224, 224),
                                 processor=None, transform=None, model_name: str = None, 
                                 target_fps: Optional[float] = None, 
                                 take_last_frames: bool = True) -> torch.Tensor:
    """Fast video preprocessing - optimized version"""
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Validate FPS
        if original_fps <= 0 or np.isnan(original_fps):
            raise ValueError(f"Invalid FPS detected: {original_fps} for video: {video_path}")
        
        # Calculate which frame indices we actually need
        if target_fps and target_fps != original_fps:
            frame_interval = max(1, int(round(original_fps / target_fps)))
            
            # For temporal consistency: calculate how many original frames we need for target_frames at target_fps
            # target_frames at target_fps = X seconds of video
            # X seconds at original_fps = X * original_fps frames
            temporal_duration_seconds = target_frames / target_fps
            needed_original_frames = int(temporal_duration_seconds * original_fps)
            
            # Ensure we don't exceed available frames
            needed_original_frames = min(needed_original_frames, total_frames)
            
            if take_last_frames:
                # Take the last N seconds of video, then downsample
                start_frame = max(0, total_frames - needed_original_frames)
                end_frame = total_frames
                original_frame_indices = list(range(start_frame, end_frame))
                # Downsample these frames to get target_fps
                available_frame_indices = original_frame_indices[::frame_interval]
            else:
                # Take the first N seconds of video, then downsample  
                original_frame_indices = list(range(min(needed_original_frames, total_frames)))
                # Downsample these frames to get target_fps
                available_frame_indices = original_frame_indices[::frame_interval]
        else:
            available_frame_indices = list(range(total_frames))
        
        # Select exactly the frames we need
        if len(available_frame_indices) >= target_frames:
            if take_last_frames:
                selected_indices = available_frame_indices[-target_frames:]
            else:
                selected_indices = available_frame_indices[:target_frames]
        else:
            # Need padding - take all available frames
            selected_indices = available_frame_indices
        
        # Debug information to verify temporal consistency
        if target_fps and len(selected_indices) > 0:
            temporal_duration = target_frames / target_fps
            duration_captured = len(selected_indices) / target_fps

            # print(f"📹 Video preprocessing: {total_frames} frames at {original_fps:.1f} FPS")
            # print(f"🎯 Target: {target_frames} frames at {target_fps} FPS = {temporal_duration:.1f} seconds")
            # print(f"📺 Actually capturing {duration_captured:.1f} seconds of video content")
            # print(f"🎯 Frame range: {selected_indices[0]} to {selected_indices[-1]} (indices)")
            # if duration_captured < temporal_duration * 0.9:
            #     print(f"⚠️  Warning: Capturing less video content than expected!")
        
        # Pre-allocate numpy array for efficiency
        frames = np.empty((len(selected_indices), target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Read only the frames we need
        frames_read = 0
        for i, frame_idx in enumerate(selected_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Batch operations: resize and color convert in one step
                frame = cv2.resize(frame, target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[i] = frame
                frames_read += 1
            else:
                # Fill with black frame if read fails
                frames[i] = np.zeros((*target_size, 3), dtype=np.uint8)
        
        # print(f"✅ Read {frames_read}/{len(selected_indices)} frames")
        
        # Handle padding if needed
        if len(selected_indices) < target_frames:
            padding_needed = target_frames - len(selected_indices)
            black_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            
            if take_last_frames:
                # Pad at beginning
                padding = np.tile(black_frame[np.newaxis], (padding_needed, 1, 1, 1))
                frames = np.concatenate([padding, frames], axis=0)
            else:
                # Pad at end  
                padding = np.tile(black_frame[np.newaxis], (padding_needed, 1, 1, 1))
                frames = np.concatenate([frames, padding], axis=0)
            
            # print(f"⚡ Padded {padding_needed} frames ({'beginning' if take_last_frames else 'end'})")
        
        # Convert to list for processor compatibility
        frames_list = [frames[i] for i in range(frames.shape[0])]
        
        # Process with model-specific processor/transform
        if processor:
            try:
                # Import here to avoid circular imports
                from train.video_training import detect_model_type
                if model_name:
                    model_info = detect_model_type(model_name)
                    if model_info.get('is_vjepa2'):
                        # V-JEPA 2 processing
                        inputs = processor(frames_list, return_tensors="pt")
                        if 'pixel_values_videos' in inputs:
                            video_tensor = inputs['pixel_values_videos'].squeeze(0)
                        elif 'pixel_values' in inputs:
                            video_tensor = inputs['pixel_values'].squeeze(0)
                        else:
                            video_tensor = list(inputs.values())[0].squeeze(0)
                    else:
                        # VideoMAE processing
                        inputs = processor(images=frames_list, return_tensors="pt")
                        video_tensor = inputs['pixel_values'].squeeze(0)
                else:
                    # Generic processing
                    inputs = processor(images=frames_list, return_tensors="pt")
                    video_tensor = inputs['pixel_values'].squeeze(0)
            except Exception as e:
                print(f"Warning: Processor failed ({e}), using manual transform")
                video_tensor = _manual_transform(frames, transform)
        else:
            video_tensor = _manual_transform(frames, transform)
        
        return video_tensor
        
    finally:
        cap.release()


def _manual_transform(frames: np.ndarray, transform=None) -> torch.Tensor:
    """Fast manual transformation using vectorized operations"""
    if transform:
        # Apply transform to each frame (unfortunately not vectorizable)
        transformed_frames = [transform(image=f)["image"] for f in frames]
        return torch.stack(transformed_frames)
    else:
        # Vectorized normalization
        frames_tensor = torch.from_numpy(frames.transpose(0, 3, 1, 2)).float() / 255.0
        return frames_tensor


# def load_full_video_frames(video_path: str, target_size: Tuple[int, int] = (224, 224), 
#                                target_fps: Optional[float] = None) -> np.ndarray:
#     """Fast loading of all video frames"""
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"Video not found: {video_path}")
    
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Cannot open video: {video_path}")
    
#     try:
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         original_fps = cap.get(cv2.CAP_PROP_FPS)
        
#         if original_fps <= 0 or np.isnan(original_fps):
#             raise ValueError(f"Invalid FPS detected: {original_fps} for video: {video_path}")
        
#         # Calculate frame sampling
#         if target_fps and target_fps != original_fps:
#             frame_interval = max(1, int(round(original_fps / target_fps)))
#             target_frame_count = total_frames // frame_interval
#         else:
#             frame_interval = 1
#             target_frame_count = total_frames
        
#         # print(f"📹 Loading {target_frame_count} frames (every {frame_interval})")
        
#         # Pre-allocate array
#         frames = np.empty((target_frame_count, target_size[1], target_size[0], 3), dtype=np.uint8)
        
#         frame_idx = 0
#         output_idx = 0
        
#         # Read frames in sequence, sampling as needed
#         while output_idx < target_frame_count:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             if frame_idx % frame_interval == 0:
#                 frame = cv2.resize(frame, target_size)
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames[output_idx] = frame
#                 output_idx += 1
            
#             frame_idx += 1
        
#         # Trim array if we read fewer frames than expected
#         if output_idx < target_frame_count:
#             frames = frames[:output_idx]
        
#         # print(f"✅ Loaded {output_idx} frames")
#         return frames
        
#     finally:
#         cap.release()

import cv2
import numpy as np
from typing import Tuple, Optional
import os

def load_full_video_frames(video_path: str, target_size: Tuple[int, int] = (224, 224), 
                          target_fps: Optional[float] = None) -> np.ndarray:
    """
    Load all video frames with accurate temporal sampling
    
    Args:
        video_path: Path to video file
        target_size: (width, height) for resizing frames
        target_fps: Target FPS for sampling. If None, uses original FPS
        
    Returns:
        np.ndarray: Array of frames with shape (num_frames, height, width, 3)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if original_fps <= 0 or np.isnan(original_fps):
            raise ValueError(f"Invalid FPS detected: {original_fps} for video: {video_path}")
        
        # Calculate video duration and target frame count
        video_duration = total_frames / original_fps
        
        if target_fps and target_fps != original_fps:
            target_frame_count = int(round(video_duration * target_fps))
            frame_interval = original_fps / target_fps
        else:
            target_frame_count = total_frames
            frame_interval = 1.0
        
        # Pre-allocate array
        frames = np.empty((target_frame_count, target_size[1], target_size[0], 3), dtype=np.uint8)
        
        output_idx = 0
        
        # Precise frame sampling
        for i in range(target_frame_count):
            frame_to_read = int(round(i * frame_interval))
            
            # Ensure we don't go beyond video length
            if frame_to_read >= total_frames:
                break
            
            # Seek to the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_read)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process and store frame
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[output_idx] = frame
            output_idx += 1
        
        # Trim array if we read fewer frames than expected
        if output_idx < target_frame_count:
            frames = frames[:output_idx]
        
        return frames
        
    finally:
        cap.release()


def apply_temperature_scaling(logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Temperature scaling consistent with training"""
    return logits / temperature