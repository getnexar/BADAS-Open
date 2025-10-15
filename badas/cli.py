#!/usr/bin/env python3
"""
Command-line interface for BADAS collision prediction.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from . import BADASModel, __version__


def predict_video():
    """CLI entry point for video prediction"""
    parser = argparse.ArgumentParser(
        description="BADAS: Predict collision risk in dashcam videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  badas-predict video.mp4
  badas-predict video.mp4 --threshold 0.7 --output results.json
  badas-predict video.mp4 --device cuda --format detailed
        """
    )
    
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("--threshold", type=float, default=0.8,
                       help="Collision probability threshold (default: 0.8)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use: cuda, cpu, or auto (default: auto)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to custom model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to JSON file")
    parser.add_argument("--format", choices=["simple", "detailed", "json"],
                       default="simple", help="Output format (default: simple)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress messages")
    
    args = parser.parse_args()
    
    # Validate input
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load model
    if not args.quiet:
        print(f"Loading BADAS model (v{__version__})...")
    
    try:
        model = BADASModel(
            device=args.device,
            confidence_threshold=args.threshold,
            checkpoint_path=args.checkpoint
        )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run inference
    if not args.quiet:
        print(f"Processing: {video_path.name}")
    
    try:
        predictions = model.predict(str(video_path))
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Format results
    results = format_results(predictions, args.threshold, args.format)
    
    # Output results
    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print(results)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results if args.format == "json" else 
                     {"predictions": predictions, "threshold": args.threshold}, 
                     f, indent=2)
        if not args.quiet:
            print(f"Results saved to: {output_path}")
    
    # Exit with appropriate code
    has_collision = any(p >= args.threshold for p in predictions)
    sys.exit(0 if not has_collision else 1)


def format_results(predictions, threshold, format_type):
    """Format prediction results based on output type"""
    
    if format_type == "json":
        high_risk = [(i * 0.125, float(p)) for i, p in enumerate(predictions) 
                    if p >= threshold]
        return {
            "total_frames": len(predictions),
            "threshold": threshold,
            "high_risk_count": len(high_risk),
            "high_risk_timestamps": high_risk,
            "max_probability": float(max(predictions)),
            "average_probability": float(sum(predictions) / len(predictions)),
            "predictions": [float(p) for p in predictions]
        }
    
    elif format_type == "detailed":
        output = []
        output.append(f"BADAS Collision Prediction Results")
        output.append("=" * 50)
        
        high_risk = []
        for i, prob in enumerate(predictions):
            timestamp = i * 0.125
            if prob >= threshold:
                high_risk.append((timestamp, prob))
                output.append(f"[{timestamp:6.1f}s] ⚠️  HIGH RISK: {prob:.2%}")
            elif prob >= 0.5:
                output.append(f"[{timestamp:6.1f}s] ⚡ Medium: {prob:.2%}")
        
        output.append("=" * 50)
        output.append(f"Summary:")
        output.append(f"  Total windows analyzed: {len(predictions)}")
        output.append(f"  High risk moments: {len(high_risk)}")
        output.append(f"  Max probability: {max(predictions):.2%}")
        output.append(f"  Average probability: {sum(predictions)/len(predictions):.2%}")
        
        if high_risk:
            output.append(f"\n🚨 COLLISION WARNING DETECTED")
            output.append(f"  First event at: {high_risk[0][0]:.1f}s")
        else:
            output.append(f"\n✅ No high collision risk detected")
        
        return "\n".join(output)
    
    else:  # simple format
        high_risk = [(i * 0.125, p) for i, p in enumerate(predictions) 
                    if p >= threshold]
        
        if high_risk:
            output = f"⚠️  COLLISION RISK DETECTED: {len(high_risk)} high-risk moments\n"
            for timestamp, prob in high_risk[:5]:  # Show first 5
                output += f"  - {timestamp:.1f}s: {prob:.2%}\n"
            if len(high_risk) > 5:
                output += f"  ... and {len(high_risk) - 5} more"
            return output.rstrip()
        else:
            return "✅ No collision risk detected"


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="badas",
        description="BADAS: Advanced Driver Assistance System CLI"
    )
    
    parser.add_argument("--version", action="version", 
                       version=f"%(prog)s {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", 
                                          help="Predict collision risk in video")
    predict_parser.add_argument("video", type=str, help="Video file path")
    predict_parser.add_argument("--threshold", type=float, default=0.8)
    predict_parser.add_argument("--device", type=str, default=None)
    
    # Info command
    info_parser = subparsers.add_parser("info", 
                                       help="Show model information")
    
    args = parser.parse_args()
    
    if args.command == "predict":
        # Redirect to predict_video with proper args
        sys.argv = ["badas-predict", args.video]
        if args.threshold != 0.8:
            sys.argv.extend(["--threshold", str(args.threshold)])
        if args.device:
            sys.argv.extend(["--device", args.device])
        predict_video()
    
    elif args.command == "info":
        print(f"BADAS v{__version__}")
        print("V-JEPA2 Based Advanced Driver Assistance System")
        print("Model: facebook/vjepa2-vitl-fpc16-256-ssv2")
        print("Input: 16 frames @ 224x224, 8 FPS")
        print("Output: Collision probability [0, 1]")
        print("\nFor more information: https://github.com/nexar-ai/badas-open")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()