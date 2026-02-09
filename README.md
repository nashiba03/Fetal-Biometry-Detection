# Fetal Biometry Detection

Deep learning system for automated detection of fetal biometry landmarks (BPD and OFD) in ultrasound images using landmark detection and segmentation approaches.

## Overview

This project implements two complementary approaches for fetal head biometry measurement from ultrasound images:

- **Task A: Landmark Detection** - Heatmap-based U-Net for direct BPD/OFD landmark localization
- **Task B: Segmentation-Based** - Cranium segmentation with ellipse fitting for biometric estimation

**This work supports AI-assisted fetal growth assessment in obstetric ultrasound and demonstrates the feasibility of low-resource CPU-based medical imaging pipelines for real-world healthcare settings.**

## Technologies Used

- **Python**, **PyTorch**
- **OpenCV**, **NumPy**, **Pandas**
- **Medical Image Processing**
- **Deep Learning (U-Net)**
- **Ultrasound Biometry Analysis**

## Project Structure
```
fetal-biometry-detection/
├── task-a-landmark-detection/     # Heatmap regression approach
│   ├── Model Weights/
│   ├── Python Script/
│   ├── Results/
│   ├── Report/
│   └── README.md
├── task-b-segmentation/           # Segmentation + ellipse fitting approach
│   ├── Model Weights/
│   ├── Python Script/
│   ├── Results/
│   ├── Report/
│   └── README.md
└── README.md                      # This file
```

## Quick Start

### Task A: Landmark Detection
```bash
cd task-a-landmark-detection
python Python\ Script/Trainer.py   # Training
python Python\ Script/Tester.py    # Inference
```
📄 [Detailed Task A Documentation](task-a-landmark-detection/README.md)

### Task B: Segmentation
```bash
cd task-b-segmentation
python Python\ Script/Trainer.py   # Training
python Python\ Script/Tester.py    # Inference
```
📄 [Detailed Task B Documentation](task-b-segmentation/README.md)

## Key Results

### Task A (Landmark Detection)
- **Final MSE**: ~0.001 (30 epochs)
- **Average Biometrics**: BPD ≈ 120 px, OFD ≈ 140 px
- **Estimated Relative Error**: ~4%
- **Architecture**: U-Net with Gaussian heatmap regression
- **Training Time**: ~2 minutes/epoch (CPU)

### Task B (Segmentation)
- **Best Dice Score**: 0.286 (18 epochs, CPU-only)
- **Output Measurements**: BPD, OFD, and HC
- **Preprocessing**: CLAHE for boundary enhancement
- **Loss Function**: Dice + Focal Loss for class imbalance
- **Improvement**: Dice 0.04 → 0.286 (consistent convergence)

## Installation
```bash
# CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Additional dependencies
pip install opencv-python numpy pandas pillow
```

## Dataset

- **622 fetal head ultrasound images**
- **Train/Validation Split**: 80% / 20%
- **Input Resolution**: 
  - Task A: 256×256
  - Task B: 384×384
- **Measurements**: Biparietal Diameter (BPD), Occipitofrontal Diameter (OFD), Head Circumference (HC)

## Technical Highlights

- **U-Net Architecture** with skip connections for both tasks
- **Heatmap Regression** (Task A): Gaussian-encoded landmarks for robust localization in noisy ultrasound data
- **Hybrid Loss Function** (Task B): Dice + Focal Loss to handle severe class imbalance
- **CPU-Optimized Pipeline**: Trained entirely on CPU with careful batch sizing and architectural choices
- **PEP-8 Compliant**: Clean, maintainable code written from scratch
- **Medical Imaging Focus**: CLAHE preprocessing for enhanced skull boundary detection

## Methodology

### Task A: Landmark Detection
- Instead of direct coordinate regression, landmarks are encoded as 2D Gaussian heatmaps
- Allows the model to learn spatial probability distributions
- More robust to speckle noise and weak edges in ultrasound images
- Euclidean distance computed between predicted landmark pairs for BPD/OFD

### Task B: Segmentation
- Deeper U-Net encoder-decoder with skip connections
- CLAHE preprocessing to enhance skull boundaries
- Ellipse fitting on segmented cranium for biometric parameter estimation
- Focal Loss addresses hard-to-segment skull regions

## Experimental Progression

**Task A:**
- Baseline CNN (direct regression): MSE ≈ 0.05
- With resizing & normalization: MSE ≈ 0.02
- **Final U-Net + heatmap regression**: MSE ≈ 0.001 ✅

**Task B:**
- Baseline (shallow U-Net): Dice ≈ 0.15
- With preprocessing & augmentation: Dice ≈ 0.22
- **Final (deeper U-Net + Focal Loss)**: Dice ≈ 0.286 ✅

## Future Work

- **GPU Acceleration**: Extended training (60-100 epochs) for higher accuracy
- **Data Augmentation**: Rotation, scaling, elastic deformation for ultrasound variability
- **Post-Processing**: CRF or morphological refinement for sharper segmentation
- **Sub-Pixel Localization**: Improved landmark precision beyond pixel-level accuracy
- **Ensemble Methods**: Integration of both approaches for robust prediction
- **Clinical Validation**: Testing against ground truth measurements from radiologists
- **Multi-Scale Attention**: Transformer-based encoders for better feature extraction

## Limitations

- **No GPU acceleration** (limited batch size and training epochs)
- **Limited data augmentation** in current implementation
- **CPU-only training** constrained experimental scale
- Task B stopped at 18 epochs due to time constraints (further training would improve Dice score)

## Results Interpretation

The achieved metrics reflect stable convergence and validate the effectiveness of:
- Preprocessing pipeline design
- Loss function choices (MSE for landmarks, Dice+Focal for segmentation)
- Architecture modifications (deeper U-Net, skip connections)

While absolute scores can be improved with GPU training, the current results demonstrate:
✅ Anatomically plausible biometric estimates  
✅ Consistent learning without overfitting  
✅ Feasibility of CPU-based medical imaging pipelines

## Citation

If you use this code, please cite:
```
Fathima Nashiba M
Fetal Biometry Detection System
B.E. Biomedical Engineering, PSG College of Technology
GitHub: https://github.com/[your-username]/fetal-biometry-detection
```

## License

MIT License (or specify your preferred license)

## Acknowledgments

This project was developed as part of the fetal biometry challenge, focusing on automated BPD and OFD landmark detection in prenatal ultrasound imaging for gestational age assessment and fetal growth monitoring.

## Author

**Fathima Nashiba M**  
B.E. Biomedical Engineering – PSG College of Technology  
**Interests**: Medical AI, Ultrasound Imaging, Deep Learning in Healthcare  

📧 [fathimanashiba03@gmail.com]  
🔗 [LinkedIn](linkedin.com/in/fathima-nashiba )  


---

⭐ **Star this repo if you find it useful!**
