"""
setup.py - Main setup script for the project
Run this first to setup the entire project
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


class ProjectSetup:
    """Setup the entire project structure and dependencies"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.colors = {
            'green': '\033[92m',
            'red': '\033[91m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'end': '\033[0m'
        }
    
    def print_colored(self, message: str, color: str = 'green'):
        """Print colored message"""
        print(f"{self.colors.get(color, '')}{message}{self.colors['end']}")
    
    def create_directory_structure(self):
        """Create project directory structure"""
        self.print_colored("\n📁 Creating directory structure...", 'blue')
        
        directories = [
            # Data directories
            "data/raw_videos",
            "data/processed",
            "data/features/audio",
            "data/features/video",
            "data/summaries",
            "data/annotations",
            
            # Source code directories
            "src/preprocessing",
            "src/feature_extraction/audio",
            "src/feature_extraction/video",
            "src/optimization",
            "src/models",
            "src/utils",
            
            # Output directories
            "outputs/videos",
            "outputs/frames",
            "outputs/logs",
            "outputs/models/checkpoints",
            "outputs/results/plots",
            
            # Config and notebooks
            "configs",
            "notebooks",
            "tests",
            "scripts",
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        init_dirs = [
            "src",
            "src/preprocessing",
            "src/feature_extraction",
            "src/feature_extraction/audio",
            "src/feature_extraction/video",
            "src/optimization",
            "src/models",
            "src/utils",
        ]
        
        for directory in init_dirs:
            init_file = Path(directory) / "__init__.py"
            init_file.touch(exist_ok=True)
        
        self.print_colored("✅ Directory structure created!", 'green')
    
    def create_requirements_file(self):
        """Create requirements.txt"""
        self.print_colored("\n📝 Creating requirements.txt...", 'blue')
        
        requirements = """# Core Dependencies
opencv-python==4.8.1.78
opencv-contrib-python==4.8.1.78
numpy==1.24.3
scipy==1.11.3

# Video/Audio Processing
moviepy==1.0.3
librosa==0.10.1
soundfile==0.12.1
pydub==0.25.1

# Deep Learning
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# Image Processing
scikit-image==0.22.0
Pillow==10.1.0

# Data Science
pandas==2.1.1
matplotlib==3.8.0
seaborn==0.13.0

# Feature Extraction
python-speech-features==0.6

# Utilities
tqdm==4.66.1
pyyaml==6.0.1
loguru==0.7.2
yt-dlp==2023.10.13

# Testing
pytest==7.4.3
"""
        
        with open("requirements.txt", 'w', encoding='utf-8') as f:
            f.write(requirements)
        
        self.print_colored("✅ requirements.txt created!", 'green')
    
    def install_dependencies(self):
        """Install Python dependencies"""
        self.print_colored("\n📦 Installing dependencies...", 'blue')
        self.print_colored("⚠️  This may take several minutes", 'yellow')
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            self.print_colored("✅ Dependencies installed successfully!", 'green')
        except subprocess.CalledProcessError:
            self.print_colored("❌ Failed to install dependencies", 'red')
            self.print_colored("Please run: pip install -r requirements.txt", 'yellow')
    
    def create_config_file(self):
        """Create default configuration file"""
        self.print_colored("\n⚙️  Creating configuration file...", 'blue')
        
        config_content = """# Configuration File
project:
  name: "Lecture Video Summarization"
  version: "1.0.0"
  author: "Mobina Abdul Rauf"

dataset:
  raw_data_path: "data/raw_videos"
  processed_path: "data/processed"
  features_path: "data/features"
  video_format: ".mp4"
  max_videos: 100
  fps: 30

preprocessing:
  color_space: "YCbCr"
  shot_threshold: 0.5
  extract_frames: true
  frame_skip: 1

features:
  video_features: ["SLBT", "LTP", "HoG", "LOOP", "LVP"]
  audio_features: ["BFCC", "MFCC", "zero_crossing", "spectral_flux", "spectral_centroid", "spectral_bandwidth"]
  mfcc_coefficients: 13
  n_fft: 2048
  hop_length: 512

model:
  name: "DRN"
  architecture:
    input_channels: 295
    num_classes: 2
    num_residual_blocks: 5
  training:
    batch_size: 16
    epochs: 50
    learning_rate: 0.001
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15

output:
  log_level: "INFO"
  save_frames: true
  save_features: true

hardware:
  device: "cuda"
  num_workers: 4
"""
        
        config_path = Path("configs/config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        self.print_colored("✅ Configuration file created!", 'green')
    
    def create_readme(self):
        """Create README file"""
        self.print_colored("\n📄 Creating README.md...", 'blue')
        
        readme_content = """# Lecture Video Summarization using Deep Learning

## Project Overview
Automated lecture video summarization using HBBEA-optimized Deep Residual Network.

## Author
**Mobina Abdul Rauf** (CMS ID: 400944)

### Committee Members
- Dr. Farzana Jabeen
- Dr. Bilal Ali

### Supervisor
- Dr. Shah Khalid

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd lecture_video_summarization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup dataset
python scripts/dataset_downloader.py
```

## Project Structure

```
lecture_video_summarization/
├── data/                  # Data directory
│   ├── raw_videos/       # Raw lecture videos
│   ├── processed/        # Processed data
│   └── features/         # Extracted features
├── src/                  # Source code
│   ├── preprocessing/    # Video preprocessing
│   ├── feature_extraction/  # Feature extraction
│   ├── optimization/     # HBBEA optimization
│   └── models/          # DRN model
├── outputs/              # Output files
├── configs/              # Configuration files
└── notebooks/            # Jupyter notebooks
```

## Usage

### 1. Download Dataset
```bash
python scripts/dataset_downloader.py
```

### 2. Load and Verify Dataset
```bash
python src/utils/video_loader.py
```

### 3. Preprocess Videos
```bash
python src/preprocessing/video_shot_segmentation.py
```

### 4. Extract Features
```bash
python src/feature_extraction/extract_all_features.py
```

### 5. Train Model
```bash
python src/models/train_drn.py
```

## Methodology

1. **Video Shot Segmentation** - YCbCr color space model
2. **Audio-Video Segmentation** - HBBEA optimization
3. **Feature Extraction**
   - Audio: BFCC, MFCC, spectral features
   - Video: SLBT, LTP, HoG, LOOP, LVP
4. **Deep Learning** - HBBEA-optimized DRN
5. **Summarization** - Merge important segments

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- NPV (Negative Predictive Value)

## References
- Kaur, P.C., & Ragha, L. (2024). Optimized deep learning enabled lecture audio video summarization. 
  Journal of Visual Communication and Image Representation, 104, 104309.

## License
Academic use only
"""
        
        with open("README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        self.print_colored("✅ README.md created!", 'green')
    
    def create_gitignore(self):
        """Create .gitignore file"""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Data
data/raw_videos/*.mp4
data/raw_videos/*.avi
data/processed/
*.h5
*.hdf5

# Models
outputs/models/*.pth
outputs/models/checkpoints/

# Logs
outputs/logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Large files
*.zip
*.tar.gz
*.rar
"""
        
        with open(".gitignore", 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        
        self.print_colored("✅ .gitignore created!", 'green')
    
    def run_setup(self):
        """Run complete setup"""
        print("=" * 70)
        self.print_colored("🚀 LECTURE VIDEO SUMMARIZATION - PROJECT SETUP", 'blue')
        print("=" * 70)
        
        # Create structure
        self.create_directory_structure()
        
        # Create files
        self.create_requirements_file()
        self.create_config_file()
        self.create_readme()
        self.create_gitignore()
        
        # Install dependencies
        print("\n" + "=" * 70)
        install = input("📦 Install dependencies now? (y/n): ").strip().lower()
        
        if install == 'y':
            self.install_dependencies()
        else:
            self.print_colored("\n⚠️  Remember to run: pip install -r requirements.txt", 'yellow')
        
        # Final message
        print("\n" + "=" * 70)
        self.print_colored("✅ PROJECT SETUP COMPLETE!", 'green')
        print("=" * 70)
        
        print("\n📋 Next Steps:")
        print("  1. pip install -r requirements.txt (if not installed)")
        print("  2. python scripts/dataset_downloader.py")
        print("  3. python src/utils/video_loader.py")
        print("\n💡 Check README.md for detailed instructions")
        print("=" * 70 + "\n")


def main():
    """Main function"""
    setup = ProjectSetup()
    setup.run_setup()


if __name__ == "__main__":
    main()