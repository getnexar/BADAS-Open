#!/usr/bin/env python3
"""
Shared video processing utilities for BADAS inference.
Self-contained - no private training module dependencies.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2

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


# ---------------------------------------------------------------------------
# Self-contained model-type detection (replaces train.video_training.detect_model_type)
# ---------------------------------------------------------------------------


def detect_model_type(model_name: str) -> Dict[str, Any]:
    """
    Detect model type and properties from model name.
    Self-contained replacement for the private training module function.
    """
    name_lower = model_name.lower()

    is_vjepa2 = "vjepa2" in name_lower or "vjepa" in name_lower
    is_videomae = "videomae" in name_lower or "video-mae" in name_lower

    # Extract crop size from model name (e.g., "vjepa2-vitl-fpc16-256" -> 256)
    crop_size = None
    if is_vjepa2:
        m = re.search(r"fpc\d+-(\d+)", name_lower)
        if m:
            crop_size = int(m.group(1))
        else:
            crop_size = 256  # default for V-JEPA2

    if is_vjepa2:
        processor_type = "auto_video"
    elif is_videomae:
        processor_type = "videomae"
    else:
        processor_type = "auto_image"

    return {
        "is_vjepa2": is_vjepa2,
        "is_videomae": is_videomae,
        "processor_type": processor_type,
        "crop_size": crop_size,
    }


# ---------------------------------------------------------------------------
# Self-contained EnhancedVideoClassifier (replaces train.video_training.EnhancedVideoClassifier)
# Matches the architecture saved in the BADAS Lightning checkpoint.
# ---------------------------------------------------------------------------


class _MLPClassifierHead(nn.Module):
    """
    MLP classification head matching the BADAS checkpoint layout.

    The checkpoint stores parameters at sequential indices:
      0  Linear(in_dim, hidden_dim)
      1  GELU  (no params)
      2  LayerNorm(hidden_dim)
      3  Dropout (no params)
      4  Linear(hidden_dim, hidden_dim)
      5  GELU  (no params)
      6  LayerNorm(hidden_dim)
      7  Dropout (no params)
      ...repeated for each hidden layer...
      N  Linear(hidden_dim, num_classes)

    This matches `head_type='mlp'` with `head_num_layers` hidden blocks.
    """

    def __init__(
        self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float = 0.1
    ):
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))  # 0, 4, ...
            layers.append(nn.GELU())  # 1, 5, ... (no params)
            layers.append(nn.LayerNorm(hidden_dim))  # 2, 6, ...
            layers.append(nn.Dropout(dropout))  # 3, 7, ... (no params)
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))  # final linear
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class _TemporalAttentionProcessor(nn.Module):
    """Temporal attention pooling over patch tokens."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) or (B, N, D)
        out, _ = self.attention(x, x, x)
        out = self.norm(out)
        return out.mean(dim=1)  # pool over sequence


class EnhancedVideoClassifier(nn.Module):
    """
    Self-contained V-JEPA2 video classifier that matches the BADAS checkpoint.

    Architecture (inferred from checkpoint state_dict):
      backbone  : VJEPA2Model (encoder + predictor from HuggingFace)
      temporal_processor : MultiheadAttention(hidden_size, num_heads) + LayerNorm
      classifier : MLP (Linear -> LayerNorm -> Linear -> LayerNorm -> Linear)
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__()
        from transformers import VJEPA2Model as _VJEPA2Model

        hidden_size = 1024  # ViT-L hidden size
        num_heads = config.get("temporal_num_heads", 8)
        head_hidden_dim = config.get("head_hidden_dim", 768)
        head_num_layers = config.get("head_num_layers", 3)
        num_classes = config.get("num_classes", 2)
        head_dropout = config.get("head_dropout", 0.1)

        self.backbone = _VJEPA2Model.from_pretrained(
            model_name,
            cache_dir=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub")),
        )
        self.temporal_processor = _TemporalAttentionProcessor(hidden_size, num_heads)
        self.classifier = _MLPClassifierHead(
            in_dim=hidden_size,
            hidden_dim=head_hidden_dim,
            num_layers=head_num_layers,
            num_classes=num_classes,
            dropout=head_dropout,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, T, C, H, W)

        Returns:
            logits: (B, num_classes)
        """
        # VJEPA2Model expects pixel_values with shape (B, T, C, H, W)
        outputs = self.backbone(pixel_values=pixel_values)
        # last_hidden_state: (B, N, D) — patch tokens from the encoder
        hidden = outputs.last_hidden_state
        pooled = self.temporal_processor(hidden)  # (B, D)
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """Get the appropriate device for model loading."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_model_file(model_path: str) -> None:
    """Validate that model file exists and is readable."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(model_path):
        raise ValueError(f"Path is not a file: {model_path}")


# ---------------------------------------------------------------------------
# Processor / transform helpers
# ---------------------------------------------------------------------------


def get_processor_for_model(model_name: str):
    """Get appropriate processor for model type."""
    model_info = detect_model_type(model_name)

    if model_info["processor_type"] == "auto_video" and HAS_AUTO_PROCESSOR:
        try:
            processor = AutoVideoProcessor.from_pretrained(model_name)
            print("Using AutoVideoProcessor for V-JEPA 2")
            return processor
        except Exception as e:
            print(f"Failed to load AutoVideoProcessor ({e})")

    elif model_info["processor_type"] == "videomae" and HAS_VIDEOMAE_PROCESSOR:
        try:
            processor = VideoMAEImageProcessor.from_pretrained(model_name)
            print("Using VideoMAEImageProcessor")
            return processor
        except Exception as e:
            print(f"Failed to load VideoMAEImageProcessor ({e})")

    elif model_info["processor_type"] == "auto_image" and HAS_AUTO_PROCESSOR:
        try:
            processor = AutoImageProcessor.from_pretrained(model_name)
            print("Using AutoImageProcessor")
            return processor
        except Exception as e:
            print(f"Failed to load AutoImageProcessor ({e})")

    print("Using manual transforms")
    return None


def get_transform_for_model(model_name: str, img_size: int = 224) -> A.Compose:
    """Get transformation pipeline for model type."""
    model_info = detect_model_type(model_name)

    # Use V-JEPA crop size if available
    if model_info["is_vjepa2"] and model_info["crop_size"]:
        img_size = model_info["crop_size"]

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_state_dict_from_lightning_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Load config and stripped state_dict from a PyTorch Lightning checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from hyper_parameters
    hp = ckpt.get("hyper_parameters", {})
    config = hp.get("model", {})

    # The PL checkpoint stores weights under 'state_dict' with 'model.' prefix
    raw_sd = ckpt.get("state_dict", ckpt.get("model_state_dict", {}))

    # Strip the leading 'model.' prefix that PL adds
    stripped_sd: Dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if k.startswith("model."):
            stripped_sd[k[len("model.") :]] = v
        else:
            stripped_sd[k] = v

    return config, stripped_sd


def load_vjepa_model(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Load V-JEPA-based EnhancedVideoClassifier.

    The checkpoint may be:
      - A plain dict with 'model' or 'model_state_dict' key, OR
      - A PyTorch Lightning checkpoint with 'state_dict' + 'hyper_parameters'

    Args:
        model_name: HuggingFace model identifier
        checkpoint_path: Path to .ckpt / .pth file
        device: Target device

    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = get_device()

    if checkpoint_path:
        validate_model_file(checkpoint_path)

    # Default config (overridden by checkpoint if available)
    config: Dict[str, Any] = {
        "num_classes": 2,
        "temporal_num_heads": 8,
        "head_hidden_dim": 768,
        "head_num_layers": 3,
        "head_dropout": 0.1,
    }

    stripped_sd: Optional[Dict[str, torch.Tensor]] = None

    if checkpoint_path:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to open checkpoint {checkpoint_path}: {e}")

        # Determine checkpoint format
        if "state_dict" in ckpt and "hyper_parameters" in ckpt:
            # PyTorch Lightning format
            hp = ckpt.get("hyper_parameters", {})
            saved_config = hp.get("model", {})
            # Merge saved config into defaults
            for key in (
                "num_classes",
                "temporal_num_heads",
                "head_hidden_dim",
                "head_num_layers",
                "head_dropout",
            ):
                if key in saved_config:
                    config[key] = saved_config[key]

            raw_sd = ckpt["state_dict"]
            # Strip 'model.' prefix added by Lightning
            stripped_sd = {}
            for k, v in raw_sd.items():
                new_key = k[len("model.") :] if k.startswith("model.") else k
                stripped_sd[new_key] = v

        elif "config" in ckpt:
            # Internal format with explicit config
            saved_config = ckpt["config"]
            for key in config:
                if key in saved_config:
                    config[key] = saved_config[key]
            raw_sd = ckpt.get("model", ckpt.get("model_state_dict"))
            if raw_sd is None:
                raise KeyError(
                    "Checkpoint missing model weights under 'model' or 'model_state_dict'"
                )
            stripped_sd = raw_sd

        elif "model" in ckpt:
            stripped_sd = ckpt["model"]
        elif "model_state_dict" in ckpt:
            stripped_sd = ckpt["model_state_dict"]
        else:
            raise KeyError(
                f"Unrecognised checkpoint format in {checkpoint_path}. "
                "Expected keys: 'state_dict'+'hyper_parameters', 'model', or 'model_state_dict'."
            )

    # Build model
    model = EnhancedVideoClassifier(model_name=model_name, config=config).to(device)

    if stripped_sd is not None:
        # The state_dict uses 'backbone.encoder.*' while EnhancedVideoClassifier
        # stores the VJEPA2Model as self.backbone — keys already match.
        missing, unexpected = model.load_state_dict(stripped_sd, strict=False)
        if missing:
            print(f"Warning: missing keys when loading checkpoint ({len(missing)} keys)")
        if unexpected:
            print(f"Warning: unexpected keys in checkpoint ({len(unexpected)} keys)")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Video preprocessing
# ---------------------------------------------------------------------------


def preprocess_video_frames(
    video_path: str,
    target_frames: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    processor=None,
    transform=None,
    model_name: str = None,
    target_fps: Optional[float] = None,
    take_last_frames: bool = True,
) -> torch.Tensor:
    """Fast video preprocessing."""

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

        if target_fps and target_fps != original_fps:
            frame_interval = max(1, int(round(original_fps / target_fps)))
            temporal_duration_seconds = target_frames / target_fps
            needed_original_frames = int(temporal_duration_seconds * original_fps)
            needed_original_frames = min(needed_original_frames, total_frames)

            if take_last_frames:
                start_frame = max(0, total_frames - needed_original_frames)
                original_frame_indices = list(range(start_frame, total_frames))
                available_frame_indices = original_frame_indices[::frame_interval]
            else:
                original_frame_indices = list(range(min(needed_original_frames, total_frames)))
                available_frame_indices = original_frame_indices[::frame_interval]
        else:
            available_frame_indices = list(range(total_frames))

        if len(available_frame_indices) >= target_frames:
            if take_last_frames:
                selected_indices = available_frame_indices[-target_frames:]
            else:
                selected_indices = available_frame_indices[:target_frames]
        else:
            selected_indices = available_frame_indices

        frames = np.empty(
            (len(selected_indices), target_size[1], target_size[0], 3), dtype=np.uint8
        )

        for i, frame_idx in enumerate(selected_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[i] = frame
            else:
                frames[i] = np.zeros((*target_size, 3), dtype=np.uint8)

        if len(selected_indices) < target_frames:
            padding_needed = target_frames - len(selected_indices)
            black_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            if take_last_frames:
                padding = np.tile(black_frame[np.newaxis], (padding_needed, 1, 1, 1))
                frames = np.concatenate([padding, frames], axis=0)
            else:
                padding = np.tile(black_frame[np.newaxis], (padding_needed, 1, 1, 1))
                frames = np.concatenate([frames, padding], axis=0)

        frames_list = [frames[i] for i in range(frames.shape[0])]

        if processor:
            try:
                model_info = detect_model_type(model_name) if model_name else {}
                if model_info.get("is_vjepa2"):
                    inputs = processor(frames_list, return_tensors="pt")
                    if "pixel_values_videos" in inputs:
                        video_tensor = inputs["pixel_values_videos"].squeeze(0)
                    elif "pixel_values" in inputs:
                        video_tensor = inputs["pixel_values"].squeeze(0)
                    else:
                        video_tensor = list(inputs.values())[0].squeeze(0)
                else:
                    inputs = processor(images=frames_list, return_tensors="pt")
                    video_tensor = inputs["pixel_values"].squeeze(0)
            except Exception as e:
                print(f"Warning: Processor failed ({e}), using manual transform")
                video_tensor = _manual_transform(frames, transform)
        else:
            video_tensor = _manual_transform(frames, transform)

        return video_tensor

    finally:
        cap.release()


def _manual_transform(frames: np.ndarray, transform=None) -> torch.Tensor:
    """Manual transformation using vectorized operations."""
    if transform:
        transformed_frames = [transform(image=f)["image"] for f in frames]
        return torch.stack(transformed_frames)
    else:
        frames_tensor = torch.from_numpy(frames.transpose(0, 3, 1, 2)).float() / 255.0
        return frames_tensor


def load_full_video_frames(
    video_path: str,
    target_size: Tuple[int, int] = (224, 224),
    target_fps: Optional[float] = None,
) -> np.ndarray:
    """
    Load all video frames with accurate temporal sampling.

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

        video_duration = total_frames / original_fps

        if target_fps and target_fps != original_fps:
            target_frame_count = int(round(video_duration * target_fps))
            frame_interval = original_fps / target_fps
        else:
            target_frame_count = total_frames
            frame_interval = 1.0

        frames = np.empty((target_frame_count, target_size[1], target_size[0], 3), dtype=np.uint8)

        output_idx = 0
        for i in range(target_frame_count):
            frame_to_read = int(round(i * frame_interval))
            if frame_to_read >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_read)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[output_idx] = frame
            output_idx += 1

        if output_idx < target_frame_count:
            frames = frames[:output_idx]

        return frames

    finally:
        cap.release()


def apply_temperature_scaling(logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Temperature scaling consistent with training."""
    return logits / temperature
