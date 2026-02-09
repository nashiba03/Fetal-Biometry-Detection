Fetal Biometry Detection

AI-powered fetal ultrasound analysis system for automatic estimation of
Biparietal Diameter (BPD), Occipitofrontal Diameter (OFD), and Head Circumference (HC)
using U-Net–based landmark detection and segmentation.

Designed as a low-resource, CPU-efficient medical imaging pipeline for
AI-assisted obstetric screening and fetal growth monitoring.

Overview

This project implements two complementary approaches for fetal head biometry measurement from ultrasound images:

Task A – Landmark Detection: Heatmap-based U-Net for direct BPD/OFD landmark localization

Task B – Segmentation-Based: Cranium segmentation with ellipse fitting for biometric estimation

This work supports AI-assisted fetal growth assessment in obstetric ultrasound and demonstrates the feasibility of CPU-based medical imaging pipelines for real-world healthcare environments.

Technologies Used

Python, PyTorch

OpenCV, NumPy, Pandas

Medical Image Processing

Deep Learning (U-Net Architecture)

Ultrasound Biometry Analysis

Project Structure
fetal-biometry-detection/
├── task-a-landmark-detection/
│   ├── Model Weights/
│   ├── Python Script/
│   ├── Results/
│   ├── Report/
│   └── README.md
├── task-b-segmentation/
│   ├── Model Weights/
│   ├── Python Script/
│   ├── Results/
│   ├── Report/
│   └── README.md
└── README.md

Quick Start
Task A – Landmark Detection
cd task-a-landmark-detection
python Python\ Script/Trainer.py
python Python\ Script/Tester.py

Task B – Segmentation
cd task-b-segmentation
python Python\ Script/Trainer.py
python Python\ Script/Tester.py


Detailed documentation for each task is available inside the respective folders.

Key Results
Task A – Landmark Detection

Final training MSE after convergence

Stable prediction of BPD and OFD landmarks

Anatomically plausible biometric estimation

U-Net with Gaussian heatmap regression

CPU-efficient training pipeline

Task B – Segmentation

Consistent convergence during training

Outputs BPD, OFD, and HC measurements

CLAHE preprocessing for skull boundary enhancement

Dice + Focal Loss for class imbalance handling

CPU-only optimized deep learning workflow

Dataset

622 fetal head ultrasound images

80/20 train–validation split

Input resolution:

Task A: 256 × 256

Task B: 384 × 384

Measurements derived: BPD, OFD, HC

Technical Highlights

U-Net with skip connections for spatial precision

Heatmap regression for robust landmark localization in noisy ultrasound

Hybrid loss design for segmentation stability under class imbalance

Fully CPU-optimized training pipeline

Clean, maintainable PEP-8 compliant implementation

Medical-focused preprocessing using CLAHE

Methodology
Task A – Landmark Detection

Landmarks encoded as 2D Gaussian heatmaps instead of direct coordinates

Enables learning of spatial probability distributions

Robust against speckle noise, weak edges, and intensity variation

Euclidean distance used to compute BPD and OFD

Task B – Segmentation

Deep encoder–decoder U-Net with skip connections

CLAHE preprocessing enhances skull boundaries

Ellipse fitting applied to segmented cranium for biometric estimation

Focal Loss improves learning in difficult skull regions

Sample Outputs

Predicted landmark heatmaps for BPD and OFD

Segmented fetal skull masks

Computed biometric measurements (BPD, OFD, HC)

Example visual outputs are available in each task’s Results/ folder.

Future Work

GPU-accelerated extended training

Advanced data augmentation for ultrasound variability

Post-processing refinement for sharper segmentation

Sub-pixel landmark localization

Ensemble integration of landmark + segmentation pipelines

Clinical validation against radiologist measurements

Exploration of attention-based or transformer encoders

Limitations

Training performed on CPU-only environment

Limited augmentation in current version

Experimental scale constrained by compute resources

Segmentation training duration limited; further training expected to improve performance

Results Interpretation

The obtained performance demonstrates:

Stable model convergence

Anatomically consistent biometric predictions

Effectiveness of preprocessing, loss design, and architecture choices

Practical feasibility of low-resource medical AI deployment

Citation

If you use this work, please cite:

Fathima Nashiba M.
AI-Based Fetal Biometry Detection using U-Net.
B.E. Biomedical Engineering, PSG College of Technology, 2026.

License

MIT License

Acknowledgments

Developed as part of a fetal biometry challenge focused on automated
BPD and OFD estimation in prenatal ultrasound imaging for
gestational age assessment and fetal growth monitoring.

Author

Fathima Nashiba M
B.E. Biomedical Engineering – PSG College of Technology (2022–2026)
Interests: Medical AI • Ultrasound Imaging • Deep Learning in Healthcare

📧 fathimanashiba03@gmail.com

🔗 https://www.linkedin.com/in/your-linkedin-id

⭐ If you find this project useful, consider giving it a star.
