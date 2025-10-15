"""
Setup script for BADAS: V-JEPA2 Based Advanced Driver Assistance System
"""
import os
from setuptools import setup, find_packages

# Read the README file
def read_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Parse requirements
def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
        return [line for line in lines if line and not line.startswith('#')]

setup(
    name="badas",
    version="1.0.0",
    author="Nexar AI Research",
    author_email="research@nexar.com",
    description="State-of-the-art collision prediction for ego-centric threat detection",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/nexar-ai/badas-open",
    project_urls={
        "Bug Tracker": "https://github.com/nexar-ai/badas-open/issues",
        "Documentation": "https://nexar-ai.github.io/badas-open",
        "Source Code": "https://github.com/nexar-ai/badas-open",
        "Paper": "https://arxiv.org/abs/2025.xxxxx",
        "Dataset": "https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction",
        "Model Hub": "https://huggingface.co/nexar-ai/badas-open",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.40.0",
        "huggingface_hub>=0.20.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
        "video": [
            "moviepy>=1.0.3",
            "imageio>=2.25.0",
            "imageio-ffmpeg>=0.4.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "badas=badas.cli:main",
            "badas-predict=badas.cli:predict_video",
        ],
    },
    include_package_data=True,
    package_data={
        "badas": ["config.json", "assets/*"],
    },
    zip_safe=False,
    keywords=[
        "collision-detection", 
        "autonomous-driving", 
        "computer-vision",
        "deep-learning",
        "adas",
        "driver-assistance",
        "safety-systems",
        "v-jepa",
        "video-analysis",
        "ego-centric",
        "dashcam",
        "pytorch",
    ],
)