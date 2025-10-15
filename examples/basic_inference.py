#!/usr/bin/env python3
"""
Basic inference example for BADAS collision prediction model.

This script demonstrates how to:
1. Load the BADAS model
2. Run inference on a video file
3. Interpret the collision predictions
"""

import argparse
import sys
import os
from pathlib import Path

# Try to import BADAS - both from installed package and local development
try:
    from badas import BADASModel
except ImportError:
    # Try to import from parent directory (for development)
    parent_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(parent_dir))
    try:
        from badas import BADASModel
        print("Note: Using BADAS from local directory (development mode)")
    except ImportError:
        print("Error: Could not import BADAS.")
        print("Please either:")
        print("  1. Install it: pip install badas")
        print("  2. Run from repository: pip install -e .")
        print("  3. Ensure the script is in the examples/ directory")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run BADAS collision prediction on a video")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("--threshold", type=float, default=0.8, 
                       help="Collision probability threshold (default: 0.8)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu, default: auto)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to custom checkpoint file (optional)")
    
    args = parser.parse_args()
    
    # Validate input
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"Loading BADAS model...")
    model = BADASModel(
        device=args.device,
        confidence_threshold=args.threshold,
        checkpoint_path=args.checkpoint
    )
    
    print(f"Processing video: {video_path}")
    print(f"Threshold: {args.threshold}")
    print("-" * 50)
    
    # Run inference
    try:
        predictions = model.predict(str(video_path))
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)
    
    # Analyze results
    print(f"\nResults:")
    print(f"Total predictions: {len(predictions)}")
    
    # Find high-risk moments
    high_risk_frames = []
    for i, prob in enumerate(predictions):
        timestamp = i * 0.125  # 8 FPS
        
        if prob >= args.threshold:
            high_risk_frames.append((timestamp, prob))
            print(f"⚠️  High collision risk at {timestamp:.1f}s: {prob:.2%}")
        elif prob >= 0.5:
            print(f"⚡ Medium risk at {timestamp:.1f}s: {prob:.2%}")
    
    # Summary
    print("\n" + "=" * 50)
    if high_risk_frames:
        print(f"🚨 COLLISION WARNING: {len(high_risk_frames)} high-risk moments detected!")
        
        # Estimate time to first collision
        first_collision_time = high_risk_frames[0][0]
        print(f"First high-risk event at: {first_collision_time:.1f} seconds")
        
        # Find peak risk
        peak_risk = max(high_risk_frames, key=lambda x: x[1])
        print(f"Peak collision probability: {peak_risk[1]:.2%} at {peak_risk[0]:.1f}s")
    else:
        print("✅ No high collision risk detected in this video")
    
    # Additional statistics
    avg_risk = sum(predictions) / len(predictions)
    max_risk = max(predictions)
    print(f"\nStatistics:")
    print(f"  Average risk: {avg_risk:.2%}")
    print(f"  Maximum risk: {max_risk:.2%}")


if __name__ == "__main__":
    main()