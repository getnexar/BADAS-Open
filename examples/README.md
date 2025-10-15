# BADAS Examples

This directory contains example notebooks and scripts demonstrating how to use the BADAS collision prediction model.

## 🚀 Quick Start (Without Installation)

You can run these examples **without installing the package** via pip. The examples automatically detect and import from the local repository:

```bash
# Clone the repository
git clone https://github.com/nexar-ai/badas-open.git
cd badas-open/examples

# Test that imports work
python test_import.py

# Run basic inference
python basic_inference.py your_video.mp4

# Or start Jupyter for notebooks
jupyter notebook
```

## 📚 Available Examples

### 1. **visualization_demo.ipynb** 📊
Interactive notebook for collision risk analysis and visualization.

**Features:**
- Load and configure BADAS model
- Process videos with sliding window prediction
- Apply temporal smoothing for stable predictions
- Generate comprehensive visualizations:
  - Time-series plots (raw vs smoothed predictions)
  - Risk distribution charts
  - Probability histograms
  - Risk timeline heatmaps
- Export results (JSON, CSV)

**Use Cases:**
- Analyze dashcam footage for safety assessment
- Generate risk reports for fleet management
- Research and development testing

### 2. **video_overlay_demo.ipynb** 🎬
Create videos with visual risk overlays burned into the frames.

**Features:**
- Real-time risk visualization on video
- Customizable overlay components:
  - Circular risk gauge with percentage
  - Color-coded borders (green/yellow/red)
  - Risk timeline bar
  - Timestamp display
- Multiple style presets (minimal, cinematic, default)
- Batch video processing
- Side-by-side comparison views

**Use Cases:**
- Create demonstration videos
- Training and education materials
- Visual validation of model predictions
- Safety review presentations

### 3. **basic_inference.py** 💻
Command-line script for quick video analysis.

**Usage:**
```bash
python basic_inference.py video.mp4 --threshold 0.8 --device cuda
```

**Options:**
- `--threshold`: Risk threshold for warnings (default: 0.8)
- `--device`: Computing device (cuda/cpu, auto-detected)
- `--checkpoint`: Path to custom model weights (optional)

**Output:**
- Frame-by-frame risk assessment
- Statistical summary
- High-risk moment detection

### 4. **test_import.py** 🔧
Utility script to verify BADAS imports are working correctly.

```bash
python test_import.py
```

## 📋 Requirements

### Minimum Requirements
```bash
pip install torch torchvision opencv-python numpy
```

### For Full Notebook Features
```bash
pip install matplotlib seaborn pandas tqdm jupyter ipywidgets
```

### For Video Overlay Generation
```bash
pip install opencv-python pillow moviepy
```

## 🎯 Usage Patterns

### Development Mode (No Installation)
The examples automatically add the parent directory to the Python path, allowing you to use BADAS without installation:

```python
# This works automatically in the examples
from badas import BADASModel, preprocess_video
```

### Installed Package Mode
If you've installed BADAS via pip:

```bash
pip install badas
# or for development
pip install -e .
```

Then the imports work directly without path manipulation.

## 💡 Tips

1. **GPU Acceleration**: The examples automatically detect and use CUDA if available
2. **Memory Management**: For long videos, use larger stride values (4-8) to reduce memory usage
3. **Batch Processing**: Process multiple videos efficiently using the batch functions
4. **Custom Models**: Load your own trained checkpoints using the `checkpoint_path` parameter

## 🔍 Troubleshooting

### Import Errors
If you get import errors:
1. Ensure you're in the `examples/` directory
2. Run `python test_import.py` to diagnose the issue
3. Check that the parent directory contains the `badas/` package

### CUDA/GPU Issues
- The models work on CPU but are much faster on GPU
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- For CPU-only: set `device='cpu'` in the examples

### Video Codec Issues
- If overlay videos don't play, try different codecs in the config
- Install ffmpeg for better codec support: `apt-get install ffmpeg`

## 📖 Learn More

- [Main README](../README.md) - Project overview and installation
- [API Documentation](https://nexar-ai.github.io/badas-open) - Detailed API reference
- [Paper](https://arxiv.org/abs/2025.xxxxx) - Technical details about the model

## 📧 Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/nexar-ai/badas-open/issues)
- Contact: research@nexar.com

---

Happy collision prediction! 🚗💨