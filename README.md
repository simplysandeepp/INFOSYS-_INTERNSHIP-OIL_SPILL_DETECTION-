# 🛢️ Oil Spill Detection System

[![Live Demo](https://img.shields.io/badge/🚀%20Try%20Now-Live%20Demo-FF4B4B?style=for-the-badge)](https://hydrovexel.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-97%25-success?style=for-the-badge)](https://github.com/simplysandeepp/oil-spill-detection-)

> **AI-powered oil spill detection with 97% accuracy. Upload any ocean image and get instant results!**

🌐 **[Try it Live - No Installation Needed!](https://hydrovexel.streamlit.app/)**

---

## 📖 Table of Contents

- [What This Does](#what-this-does)
- [Quick Start](#quick-start)
- [Complete Guide: Dataset to Deployment](#complete-guide-dataset-to-deployment)
  - [Phase 1: Dataset Preparation](#phase-1-dataset-preparation)
  - [Phase 2: Data Loading & Preprocessing](#phase-2-data-loading--preprocessing)
  - [Phase 3: Model Architecture](#phase-3-model-architecture)
  - [Phase 4: Training Process](#phase-4-training-process)
  - [Phase 5: Evaluation & Results](#phase-5-evaluation--results)
  - [Phase 6: Deployment](#phase-6-deployment)
- [Performance](#performance)
- [Visualizations](#visualizations)
- [Tech Stack](#tech-stack)
- [Author](#author)

## ✨ What This Does

Upload any satellite or aerial image → AI detects oil spills → Get detailed analysis instantly!

*Perfect for: Students, researchers, government agencies, environmental organizations, and anyone concerned about ocean safety.*

## 🚀 Quick Start

### Try Online (Easiest!)
[https://hydrovexel.streamlit.app/](https://hydrovexel.streamlit.app/)

### Run Locally

## 📚 Complete Guide: Dataset to Deployment

### Phase 1: Dataset Preparation

- **Download:** Training, validation, and test images and masks from [Zenodo Oil Spill Dataset](https://zenodo.org/).
- **Directory Structure:**

### Phase 2: Data Loading & Preprocessing

- Use Google Colab or local Python.
- Resize images to 256×256 pixels and normalize [0, 1].
- Masks: Binarize with threshold at 0.5.

### Phase 3: Model Architecture

- **U-Net** with attention gates and residual connections.
- **Parameters:** ~31 million.
- Model weights stored on Google Drive (download via `gdown` in code).

### Phase 4: Training Process

- **Platform:** Google Colab (T4 GPU)
- **Batch size:** 8 (can be adjusted by RAM)
- **Epochs:** 30
- **Learning Rate:** 0.0001 (with optional scheduler)
- **Losses:** Binary Cross-Entropy + Dice Loss

### Phase 5: Evaluation & Results

- Metrics: Accuracy, Dice coefficient, IoU, Precision, Recall.
- Visual analytics: Confusion matrix, overlays, heatmaps.

### Phase 6: Deployment

- Automatic model downloading (`gdown`) in Streamlit app.
- User-facing: Simple web UI for image upload and detection.

## ⚙️ Installation & Setup


## 🏆 Performance

| Metric          | Value | Explanation                    |
|-----------------|-------|-------------------------------|
| Accuracy        | 97%   | Correctly classified pixels   |
| Dice Coefficient| 0.95  | 95% mask overlap              |
| IoU             | 0.89  | 89% boundary detection        |
| Precision       | 96%   | Spot detection correctness    |
| Recall          | 94%   | Detects 94% of actual spills  |

## 📊 Visualizations

- Original image, ground truth mask, predicted mask
- Heatmaps highlighting oil spill areas
- Performance plots: loss curves, accuracy, IoU over epochs
 ---
# 🌊 HydroVexel: User Workflow

## 1. Access the Website
Go to [https://hydrovexel.streamlit.app/](https://hydrovexel.streamlit.app/) using any web browser.  
Instantly see a friendly introduction explaining the purpose: rapid, AI-powered oil spill detection from satellite/aerial imagery.

## 2. Read About Oil Spills and AI Benefits
The landing page educates users about:
- Oil spill risks
- Why early detection matters
- How AI enables fast analysis for environmental protection

## 3. Upload an Image for Analysis
Users can upload **satellite or aerial images** in formats like **JPG, PNG** (up to **200MB** per file).  
Drag and drop the file onto the upload area or use the **Browse files** button.

## 4. Set Analysis Options
- **Confidence Threshold Slider:** Adjust the confidence level for detection results (e.g., 0.5 for standard sensitivity).  
- **Overlay Transparency Slider:** Change how visible the AI overlay is on your detection results for easier interpretation.

## 5. Run Detection
Click **“DETECT OIL SPILL”** to start the analysis.  
The deep learning model processes the image to identify possible oil spills.  
The system rapidly scans and segments suspected spill regions.

## 6. View Your Results
Detection results are shown on the screen:

| Display Type | Description |
|---------------|-------------|
| Detection Overlay | Color-coded region indicating detected spill |
| Confidence Heatmap | Visualizes AI’s confidence per pixel |
| Binary Mask | Black-and-white region marking spill/no-spill |
| Coverage % / Stats | Shows detected area coverage, average and max confidence, number of relevant pixels |

## 7. Explore Previous Detections and Gallery
Scroll down to see detection history from your session and the database:
- All analyzed images
- Coverage stats, dates, and detection rate  
Browse a **Detection Image Gallery** of previous results, with a filter-by-date option.

## 8. Download or View Full Images
Open full **overlay**, **heatmap**, or **mask** images directly in the browser.  
Use provided links to view each processed image at full size.

## 9. Clear Results or Start Over
Click **“CLEAR RESULTS”** to reset and analyze new images.

## 10. Credits and Project Details
Learn about the project creator *(Sandeep Prajapati)*.  
See the technology stack (**Deep Learning**, **Streamlit**, **Supabase**) and access links to profiles for additional info or networking.

## 11. Purpose and Compliance
**Disclaimer:** For environmental monitoring and research only.


## 💻 Tech Stack

- **Deep Learning:** TensorFlow, Keras
- **Data Processing:** NumPy, OpenCV, PIL
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit, gdown


## 👨‍💻 Author

**Sandeep Prajapati**  
Infosys Springboard Intern   
[Github: simplysandeepp](https://github.com/simplysandeepp)  
[Email: sandeepprajapati2608@gmail.com](mailto:sandeepprajapati2608@gmail.com)  
Website: [sandeepp.in](https://sandeepp.in/)

---

*For details, sample images, or to contribute, see the repository and get in touch!*
