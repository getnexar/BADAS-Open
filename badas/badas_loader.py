import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
from huggingface_hub import hf_hub_download, snapshot_download

def load_badas_model(device: Optional[str] = None, 
                    checkpoint_path: Optional[str] = None,
                    download_weights: bool = True):
    """
    Load the BADAS collision prediction model.
    
    Args:
        device: Device to load model on ('cuda', 'cpu', or None for auto-detect)
        checkpoint_path: Path to local checkpoint file (optional)
        download_weights: Whether to download weights from HuggingFace if not found locally
    
    Returns:
        Loaded BADAS model instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Import from package structure
    from .models.vjepa import VJEPAModel
    
    # Handle checkpoint path
    if checkpoint_path is None and download_weights:
        print("Downloading BADAS model weights...")
        checkpoint_path = hf_hub_download(
            repo_id="nexar-ai/badas-open",
            filename="weights/badas_open.pth"
        )
    
    model = VJEPAModel(
        model_name="facebook/vjepa2-vitl-fpc16-256-ssv2",
        checkpoint_path=checkpoint_path,
        frame_count=16,
        img_size=224,
        window_stride=1,
        target_fps=8.0,
        use_sliding_window=True,
        device=device
    )
    
    if checkpoint_path:
        model.load()
    
    return model


class BADASModel:
    """High-level BADAS model wrapper for easy inference"""
    
    def __init__(self, device: Optional[str] = None, 
                 confidence_threshold: float = 0.8,
                 checkpoint_path: Optional[str] = None):
        """
        Initialize BADAS model.
        
        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto)
            confidence_threshold: Threshold for collision detection (0-1)
            checkpoint_path: Optional path to local checkpoint
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.model = load_badas_model(device=self.device, checkpoint_path=checkpoint_path)
    
    def predict(self, video_path: str) -> List[float]:
        """
        Predict collision probability for each frame window in video.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            List of collision probabilities for each frame window
        """
        frames = preprocess_video(video_path, target_fps=8, num_frames=16)
        
        with torch.no_grad():
            predictions = self.model.predict(frames)
        
        return predictions
    
    def estimate_time_to_accident(self, collision_probs: List[float], 
                                 fps: float = 8.0) -> Optional[float]:
        """
        Estimate time to potential collision based on probability sequence.
        
        Args:
            collision_probs: Sequence of collision probabilities
            fps: Frames per second of prediction sequence
            
        Returns:
            Estimated seconds to collision, or None if no high risk detected
        """
        # Find first frame exceeding threshold
        for i, prob in enumerate(collision_probs):
            if prob >= self.confidence_threshold:
                return i / fps
        return None
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Direct model inference on preprocessed frames"""
        return self.model(frames)


def preprocess_video(video_path: str, 
                    target_fps: float = 8.0,
                    num_frames: int = 16,
                    img_size: int = 224) -> torch.Tensor:
    """
    Preprocess video for BADAS model input.
    
    Args:
        video_path: Path to video file
        target_fps: Target frames per second for sampling
        num_frames: Number of frames to extract
        img_size: Size to resize frames to
        
    Returns:
        Preprocessed video tensor ready for model input
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample
    if target_fps and fps > target_fps:
        frame_interval = int(fps / target_fps)
    else:
        frame_interval = 1
    
    frames = []
    frame_idx = 0
    
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            # Resize and normalize frame
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    
    # Pad with last frame if needed
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    # Convert to tensor
    frames_tensor = torch.tensor(np.array(frames[:num_frames]))
    frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]
    frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension
    
    return frames_tensor
