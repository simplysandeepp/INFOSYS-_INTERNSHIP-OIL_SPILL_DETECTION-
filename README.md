# 🌊 HydroVexel - AI-Powered Oil Spill Detection System

[![Live Demo](https://img.shields.io/badge/🚀%20Try%20Now-Live%20Demo-FF4B4B?style=for-the-badge)](https://hydrovexel.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **AI Eyes Safeguarding Our Precious Oceans ✨**

An advanced deep learning system that detects and analyzes oil spills from satellite and aerial imagery with **94.57% accuracy**. Built with state-of-the-art U-Net architecture, attention mechanisms, and deployed as an interactive web application.

---

## 🌐 **[Try HydroVexel Live ](https://hydrovexel.streamlit.app/)**
🎥 **Explanatory Video:** [Click Here to Watch](https://drive.google.com/drive/folders/1cru23V5h5avZuVgT1UBg6zuYQGnc94WT?usp=drive_link)

[![🚀 Explanatory Video](https://img.shields.io/badge/🚀%20Explanatory%20Video-Click%20Me-FF4B4B?style=for-the-badge)](https://drive.google.com/drive/folders/1cru23V5h5avZuVgT1UBg6zuYQGnc94WT?usp=drive_link)



<img src="https://user-images.githubusercontent.com/74038190/235224431-e8c8c12e-6826-47f1-89fb-2ddad83b3abf.gif" width="300">
<br><br>

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Module Implementation](#-module-implementation)
- [Model Performance](#-model-performance)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Results & Visualizations](#-results--visualizations)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [About the Developer](#-about-the-developer)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🎯 Project Overview

**HydroVexel** is an AI-driven system designed to identify and monitor oil spills in marine environments using satellite imagery. The system leverages cutting-edge deep learning technologies to provide real-time detection, segmentation, and analysis of oil spills, enabling rapid response and minimizing environmental damage.

### Why This Matters

Oil spills pose severe threats to:
- 🐋 **Marine Ecosystems** - Devastating impact on aquatic life
- 🏖️ **Coastal Regions** - Contamination of beaches and shorelines
- 💰 **Local Economies** - Damage to fishing and tourism industries
- 🌍 **Global Environment** - Long-term ecological consequences

Traditional detection methods are slow, labor-intensive, and often delayed. **HydroVexel** provides instant, accurate detection to enable immediate intervention.

---

## 🚨 Problem Statement

Oil spills pose a severe threat to marine ecosystems, coastal regions, and local economies. Traditional detection methods, such as manual inspection of satellite images or physical patrolling, are:
- ⏰ **Time-consuming** - Hours to days for analysis
- 👷 **Labor-intensive** - Requires expert human inspection
- 🐌 **Often delayed** - Critical response time lost

**Solution:** Develop an AI-powered oil spill detection system using machine learning and satellite imagery to identify and localize oil spills efficiently and accurately, facilitating early intervention and supporting emergency response efforts.

---

## ✨ Key Features

### 🎯 Core Capabilities
- **Real-time Oil Spill Detection** - Upload images and get instant analysis
- **High Accuracy Segmentation** - 94.57% accuracy with U-Net + Attention architecture
- **Confidence Mapping** - Visual heatmaps showing detection confidence levels
- **Multi-format Support** - JPG, PNG satellite/aerial imagery
- **Cloud Storage Integration** - Automatic saving to Supabase database
- **Historical Analysis** - View past detections with date-based filtering

### 🖥️ User Interface
- **Interactive Web Application** - Built with Streamlit
- **Beautiful Ocean Theme** - Immersive background video with glassmorphism design
- **Real-time Visualization** - Detection overlays, heatmaps, and binary masks
- **Export Capabilities** - Download results as PNG, CSV, JSON
- **Responsive Design** - Works seamlessly on desktop and mobile

### 📊 Analytics & Reporting
- **Detection Metrics** - Coverage %, confidence scores, pixel counts
- **Image Gallery** - Browse all processed images with filters
- **Database Dashboard** - View detection statistics and trends
- **Export Options** - Full data export for further analysis

---

## 🛠️ Technology Stack

### Deep Learning & AI
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.15-D00000?style=flat-square&logo=keras&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

### Data Processing & Analysis
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=flat-square&logo=pandas&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

### Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-11557c?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-9cf?style=flat-square)

### Web Development & Deployment
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?style=flat-square&logo=supabase&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)

---

## 🏗️ Project Architecture

```
Input Image → Preprocessing → U-Net Model → Segmentation → Post-processing → Results
                                    ↓
                           Attention Gates
                           Residual Blocks
                                    ↓
                          Binary Mask + Confidence Map
                                    ↓
                        Visualization + Database Storage
```

---

## 📚 Module Implementation

### Module 1: Data Collection 📥

**Objective:** Acquire and organize satellite imagery datasets for training and evaluation.

**What I Did:**
- Sourced Oil Spill Detection Dataset from [Zenodo](https://zenodo.org/records/10555314)
- Organized data into train/validation/test splits with corresponding segmentation masks
- Configured GPU environment with TensorFlow 2.19 and mixed precision training.

**Results:**
```
✓ GPU Configured: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
✓ TensorFlow Version: 2.19.0
✓ Mixed Precision: float16

DIRECTORY VERIFICATION:
✓ Train Images: 811 files
✓ Train Masks: 811 files
✓ Val Images: 203 files
✓ Val Masks: 203 files
✓ Test Images: 254 files
✓ Test Masks: 254 files
```

**Configuration:**
- Image Size: 256×256 pixels
- Batch Size: 8
- Epochs: 30
- Learning Rate: 0.0001
- Mixed Precision: Enabled for faster training

---

### Module 2: Data Exploration & Preprocessing 🔍

**Objective:** Analyze, visualize, and prepare data for optimal model training.

**What I Did:**
- Performed statistical analysis on image properties (brightness, contrast, coverage)
- Normalized pixel values to [0, 1] range for stable training
- Applied SAR-specific speckle noise reduction filters
- Implemented advanced data augmentation (flipping, rotation, scaling, brightness adjustments)

**Results:**
```
✓ Loaded 811 training image-mask pairs
✓ Loaded 203 validation image-mask pairs
✓ Loaded 254 test image-mask pairs
✓ TensorFlow datasets created with advanced augmentation
```

**Visualizations Generated:**
1. Total Distribution Analysis
2. Sample Image-Mask Pairs
3. Oil Spill Coverage Distribution
4. Brightness Distribution
5. Contrast Distribution

---

### Module 3: Model Development 🧠

**Objective:** Design and implement a powerful deep learning architecture for precise oil spill segmentation.

**Architecture:** Enhanced U-Net with Attention Gates + Residual Connections

| Aspect | Details |
|--------|---------|
| **Model Type** | U-Net (4-Level) with Attention + Residual Connections |
| **Input Shape** | 256×256×3 (multi-channel SAR/satellite images) |
| **Output Shape** | 256×256×1 (binary mask for oil spill regions) |
| **Total Parameters** | 33,487,621 |
| **Encoder Filters** | 64 → 128 → 256 → 512 |
| **Bridge** | 1024 filters |
| **Decoder Levels** | 4 levels with Attention Gates |
| **Skip Connections** | Yes (Attention-enhanced) |
| **Residual Blocks** | Yes (all convolutional blocks) |
| **Dropout** | Applied in encoder levels 2–4 |
| **Total Layers** | 122 |
| **Loss Function** | Combined Binary Cross-Entropy + Dice Loss |
| **Metrics** | Accuracy, Dice Coefficient, IoU, Precision, Recall |

**Key Innovations:**
- **Attention Gates:** Focus on relevant features, suppress noise
- **Residual Connections:** Enable deeper networks without vanishing gradients
- **Skip Connections:** Preserve spatial information across encoder-decoder
- **Dropout Regularization:** Prevent overfitting

---

### Module 4: Training & Evaluation 🏋️

**Objective:** Train the model with optimized hyperparameters and validate performance.

**Training Configuration:**
```
Epochs: 30
Batch Size: 8
Initial Learning Rate: 0.0001
Optimizer: AdamW with weight decay
Warmup Phase: 5 epochs (prevents early instability)
Early Stopping Patience: 20 epochs
Monitoring Metric: Dice Coefficient
```

**Advanced Callbacks:**
1. Warmup Scheduler - Gradual learning rate increase
2. Model Checkpoint - Saves best performing model
3. ReduceLR on Plateau - Adaptive learning rate (patience=7)
4. Progress Monitor - Tracks improvement trends
5. Learning Rate Logger - Records LR changes
6. TensorBoard - Visual monitoring
7. Early Stopping - Patience=20, min_delta=0.0001

**Training Progress:**

| Phase | Epochs | Val Dice Score | Improvement |
|-------|--------|----------------|-------------|
| Warmup Start | Epoch 1 | 0.49 | — |
| Mid Warmup | Epoch 3 | 0.68 | +0.19 |
| Warmup End | Epoch 5 | 0.77 | +0.28 |
| Training Mid | Epoch 10 | 0.85 | +0.36 |
| Training Peak | Epoch 18 | 0.8968 | +0.40 |
| **Final (Epoch 22)** | **0.8984** | **✅ Best Score** |

**Performance Graphs Generated:**
- Loss Curve (Training vs Validation)
- Accuracy Progression
- Dice Coefficient Evolution
- IoU Tracking
- Precision & Recall Curves
- Learning Rate Schedule

---

### Module 5: Visualization of Results 🎨

**Objective:** Generate comprehensive visual outputs for model evaluation and reporting.

**Evaluation Results (Validation Set):**
```
🧠 Model Performance Summary
Accuracy:    94.57%
Precision:   96.22%
Recall:      94.69%
F1-Score:    95.45%
Specificity: 94.38%

Confusion Matrix:
True Positives (TP):  7,580,602
True Negatives (TN):  5,000,558
False Positives (FP):   297,551
False Negatives (FN):   425,097
```

**Visualization Theme:**
- **Confidence Map:** YlOrRd colormap (yellow→red = increasing confidence)
- **Confusion Matrix:** YlOrBr (counts), Reds (normalized)
- **Quality Heatmap:** RdYlGn (Red=Low IoU, Green=High IoU)

**Generated Visualizations:**
1. **Predictions with Overlays** - Original, Ground Truth, Prediction, Confidence, Overlay
2. **Confusion Matrices** - Absolute counts & normalized percentages
3. **Quality Heatmap** - IoU-based segmentation quality analysis
4. **Best vs Worst Predictions** - Top 5 best & bottom 5 worst segmentations

**Key Insights:**
- ✅ Achieved 94.57% overall accuracy
- ✅ 96% precision → Very low false alarm rate
- ✅ F1-Score 95.45% → Excellent balance
- ✅ High segmentation quality suitable for real-world deployment

---

### Module 6: Deployment via Streamlit App 🚀

**Objective:** Create an interactive, user-friendly web application for real-time oil spill detection accessible to anyone worldwide.

#### Model Deployment Strategy
- **Format:** Converted to `.h5` (Keras HDF5 format, ~400MB)
- **Storage:** Google Drive for cloud-based access
- **Download:** Automatic download via `gdown` library on first run
- **Reason:** Model too large for GitHub; enables efficient cloud deployment

#### Web Application Stack

**Frontend Framework:** Streamlit 1.37.1
- Simple, intuitive interface for non-technical users
- Real-time interaction and instant feedback
- Built-in widgets for file upload and parameter control

**Backend Processing:** Python + TensorFlow
- Model inference and prediction pipeline
- Image preprocessing and post-processing
- Metrics calculation and visualization generation

**Database & Storage:** Supabase
- PostgreSQL database for detection records
- Cloud storage for processed images
- Public URL generation for image sharing

**Visual Design:**
- **Ocean Theme:** Immersive `ocean.mp4` background video
- **Glassmorphism UI:** Semi-transparent cards with blur effects
- **Color Scheme:** Cyan/teal gradients for ocean aesthetic
- **Animations:** Smooth transitions and hover effects

#### Application Features

**1. Image Upload Interface**
- Drag-and-drop or browse file selection
- Supports JPG, PNG formats
- File size limit: Up to 200MB
- Real-time validation

**2. Interactive Controls**
- **Confidence Threshold Slider:** Adjust sensitivity (0.0 - 1.0)
- **Overlay Transparency Slider:** Control overlay visibility
- **Detect Button:** Trigger analysis
- **Clear Button:** Reset results

**3. Results Display**
- **Detection Overlay:** Color-coded spill regions on original image
- **Confidence Heatmap:** Visual representation of model confidence
- **Binary Mask:** Black & white segmentation output
- **Metrics Dashboard:**
  - Coverage percentage
  - Average confidence
  - Maximum confidence
  - Detected pixel count

**4. Advanced Visualizations**
- **Tabbed Interface:** Binary Mask, Raw JSON Data, Analysis Summary
- **Comparison Views:** Side-by-side original and processed images
- **Export Options:** Download PNG, CSV, JSON formats

**5. Database Integration**
- **Live Session Database:** Track current session detections
- **Cloud Database:** Store all detections permanently
- **Image Gallery:** Browse past detections with date filters
- **Statistics Dashboard:** Overall performance metrics

**6. Historical Analysis**
- Date-based filtering
- View all detections or filter by specific dates
- Export detection records for external analysis
- Image gallery with thumbnails and full-size viewing

#### File Architecture & Workflow

**Configuration Layer (`config/`)**
- `config.py` - Centralized settings
  - Image dimensions (256×256)
  - File paths and directories
  - Confidence thresholds
  - Visualization parameters

**Model Layer (`models/`)**
- `model_architecture.py` - U-Net architecture definition with attention gates
- `best_model.h5` - Trained model weights (auto-downloaded from Google Drive)
- `__init__.py` - Module initialization

**Utility Layer (`utils/`)**
- `inference.py` - Model loading, prediction pipeline, Google Drive integration
- `preprocessing.py` - Image loading, resizing, normalization, validation
- `visualization.py` - Overlay creation, heatmaps, confidence maps, metric displays
- `db.py` - Supabase database operations, image storage, data retrieval

**Frontend Layer**
- `streamlit_app.py` - Main application entry point
  - User interface components
  - Result visualization logic
  - Session state management
  - Database interaction handlers

**Styling Layer (`styles/`)**
- `custom.css` - Glassmorphism effects and ocean theme
- `styles2.css` - Additional UI enhancements and responsive design
- `ocean.mp4` - Background video for immersive experience
- `style.css` - Base styling

**Deployment Configuration**
- `.streamlit/secrets.toml` - Supabase credentials
- `requirements.txt` - Python dependencies
- `packages.txt` - System packages for Streamlit Cloud
- `.devcontainer/` - Development container configuration

#### Libraries Used

**Web Interface:**
- Streamlit - Main web framework
- HTML/CSS - Custom styling

**Deep Learning:**
- TensorFlow 2.19 - Model inference
- Keras 2.15 - High-level API

**Image Processing:**
- OpenCV 4.9 - Image manipulation
- PIL (Pillow) 10.3 - Image I/O
- NumPy 1.26 - Array operations

**Visualization:**
- Matplotlib 3.8 - Plotting
- Seaborn 0.13 - Statistical visualizations

**Database & Cloud:**
- Supabase - Cloud database and storage
- gdown 5.1 - Google Drive file downloads
- python-dotenv - Environment variable management

**Data Processing:**
- Pandas 2.2 - Data manipulation and export

#### Workflow Summary

1. **User uploads satellite/aerial image**
2. **Image validation and preprocessing** (resize, normalize)
3. **Model inference** (U-Net prediction)
4. **Post-processing** (threshold application, mask generation)
5. **Visualization creation** (overlays, heatmaps, metrics)
6. **Database storage** (metadata + images to Supabase)
7. **Results display** (interactive dashboard)
8. **Export options** (download PNG/CSV/JSON)

---

## 📊 Model Performance

### Accuracy Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 94.57% | Overall pixel-wise correctness |
| **Dice Coefficient** | 0.8984 | ~90% mask overlap with ground truth |
| **IoU** | 0.83-0.90 | Strong boundary detection |
| **Precision** | 96.22% | Very few false alarms |
| **Recall** | 94.69% | Detects 95% of actual spills |
| **F1-Score** | 95.45% | Excellent balance |
| **Specificity** | 94.38% | High true negative rate |

### Confusion Matrix Analysis

```
True Positives:  7,580,602   (Correctly identified spill pixels)
True Negatives:  5,000,558   (Correctly identified clean pixels)
False Positives:   297,551   (False alarms - 4% of predictions)
False Negatives:   425,097   (Missed spills - 5% of actual spills)
```

### Performance Highlights

✅ **High Precision (96.22%)** - Minimizes false alarms, critical for response teams
✅ **High Recall (94.69%)** - Catches 95% of actual spills
✅ **Balanced F1-Score (95.45%)** - Excellent precision-recall trade-off
✅ **Strong IoU (0.83-0.90)** - Accurate boundary delineation
✅ **Robust Specificity (94.38%)** - Reliable clean water detection

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- GPU (recommended) or CPU
- 8GB+ RAM
- Internet connection (for model download)

### Quick Start

1. **Clone the Repository**
```bash
git clone https://github.com/simplysandeepp/hydrovexel.git
cd hydrovexel
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Supabase (Optional)**
Create `.env` file:
```bash
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

5. **Run the Application**
```bash
streamlit run streamlit_app.py
```

6. **Access the App**
Open browser and navigate to: `http://localhost:8501`

### Cloud Deployment

**Streamlit Cloud (Recommended):**
1. Fork the repository
2. Connect to Streamlit Cloud
3. Add Supabase secrets in Streamlit dashboard
4. Deploy with one click

**Access Live Demo:** [https://hydrovexel.streamlit.app/](https://hydrovexel.streamlit.app/)

---

## 💡 Usage Guide

### Step-by-Step Instructions

1. **Access the Website**
   - Visit [hydrovexel.streamlit.app](https://hydrovexel.streamlit.app/)
   - No installation required!

2. **Upload an Image**
   - Click "Browse files" or drag-and-drop
   - Supported formats: JPG, PNG
   - Recommended: Satellite or aerial ocean imagery

3. **Adjust Settings (Optional)**
   - **Confidence Threshold:** Set detection sensitivity (default: 0.5)
   - **Overlay Transparency:** Adjust visualization clarity

4. **Run Detection**
   - Click "🔍 DETECT OIL SPILL" button
   - Processing typically takes 2-5 seconds

5. **View Results**
   - **Detection Overlay:** Colored mask on original image
   - **Confidence Heatmap:** AI confidence visualization
   - **Binary Mask:** Black & white segmentation
   - **Metrics Dashboard:** Coverage, confidence, pixel counts

6. **Explore Additional Features**
   - View detailed analysis in tabs
   - Download results (PNG, CSV, JSON)
   - Browse detection history
   - Access image gallery with date filters

7. **Database & Analytics**
   - View session statistics
   - Explore historical detections
   - Filter by date
   - Export data for reports

---

## 📁 Project Structure

```
HYDROVEXEL/
│
├── .devcontainer/              # Development container configuration
│   └── devcontainer.json
│
├── .streamlit/                 # Streamlit configuration
│   └── secrets.toml            # Supabase credentials (not in repo)
│
├── config/                     # Configuration files
│   ├── __init__.py
│   └── config.py               # Centralized settings
│
├── models/                     # Model architecture and weights
│   ├── __init__.py
│   ├── model_architecture.py   # U-Net implementation
│   └── best_model.h5           # Trained weights (auto-downloaded)
│
├── notebooks/                  # Training notebooks
│   └── oil_spill.ipynb         # Complete training pipeline
│
├── styles/                     # UI styling and assets
│   ├── custom.css              # Ocean theme with glassmorphism
│   ├── ocean.mp4               # Background video
│   ├── style.css               # Base styles
│   └── styles2.css             # Enhanced UI styles
│
├── temp_uploads/               # Temporary file storage
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── db.py                   # Supabase integration
│   ├── inference.py            # Model loading and prediction
│   ├── preprocessing.py        # Image preprocessing
│   └── visualization.py        # Result visualization
│
├── venv/                       # Virtual environment (local only)
│
├── .env                        # Environment variables (not in repo)
├── .gitattributes              # Git LFS configuration
├── .gitignore                  # Git ignore rules
├── packages.txt                # System packages for deployment
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── streamlit_app.py            # Main application entry point
```

---
## 🎨 Results & Visualizations

### Sample Detection Results

Our model successfully detects oil spills with high accuracy and provides multiple visualization formats:

**1. Detection Overlay**
- Original image with colored spill regions
- Red highlighting for detected oil spills
- Semi-transparent overlay for context

**2. Confidence Heatmap**
- Color-coded confidence levels
- Yellow → Orange → Red (low to high confidence)
- Helps identify high-certainty detections

**3. Binary Segmentation Mask**
- Black & white output
- White = Oil spill detected
- Black = Clean water

**4. Metrics Dashboard**
- Coverage percentage
- Average confidence score
- Maximum confidence
- Pixel counts

### Training Visualizations

**Performance Graphs:**
- Loss curves (training vs validation)
- Accuracy progression over epochs
- Dice coefficient evolution
- IoU tracking
- Precision & recall curves
- Learning rate schedule

**Evaluation Visualizations:**
- Confusion matrices (absolute & normalized)
- Quality heatmaps (IoU-based)
- Best vs worst predictions comparison
- Statistical distribution charts

---

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

**Disclaimer:** For environmental monitoring and research only.

---

# 🔮 Future Enhancements

### Planned Features

1. **Real-time Satellite Integration**
   - Direct API access to Sentinel-1, MODIS satellites
   - Automatic periodic monitoring of designated areas
   - Alert system for new spill detections

2. **Multi-temporal Analysis**
   - Track spill evolution over time
   - Predict spill movement patterns
   - Estimate cleanup progress

3. **Enhanced Model**
   - Spill severity classification (light, moderate, heavy)
   - Oil type identification
   - Spill age estimation

4. **Mobile Application**
   - Native iOS and Android apps
   - Offline detection capability
   - GPS-tagged detections

5. **Advanced Analytics**
   - Automated report generation
   - Trend analysis and predictions
   - Integration with response planning tools

6. **Collaboration Features**
   - Multi-user access with roles
   - Shared detection workspace
   - Comment and annotation system

7. **API Development**
   - RESTful API for third-party integration
   - Batch processing capabilities
   - Webhook notifications

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### How to Contribute

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Add appropriate comments and documentation
- Include tests for new features
- Update README if needed
- Ensure all tests pass before submitting

### Areas for Contribution

- 🐛 Bug fixes and issue resolution
- ✨ New feature development
- 📝 Documentation improvements
- 🎨 UI/UX enhancements
- 🧪 Testing and quality assurance
- 🌍 Internationalization (i18n)

---

## 👨‍💻 About the Developer

### Sandeep Prajapati

**AI Enthusiast | Deep Learning Engineer | Environmental Tech Advocate**

I'm passionate about leveraging artificial intelligence to solve real-world environmental challenges. HydroVexel represents my commitment to protecting our oceans through innovative technology.

### Professional Background

- 🎓 **Education:** B.Tech in Computer Science and Engineering (AI & ML)
  - Galgotias University (2023-2027)
- 💼 **Current Role:** AI Intern @ Infosys Springboard
- 🌟 **Leadership:**
  - Google Student Ambassador
  - GSSOC'25 Mentor
  - Core Member @ GDG OC GU
  - Super Contributor @ Hacktoberfest'25

### Connect With Me

[![GitHub](https://img.shields.io/badge/GitHub-simplysandeepp-181717?style=for-the-badge&logo=github)](https://github.com/simplysandeepp)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sandeep_Prajapati-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/simplysandeepp/)
[![Email](https://img.shields.io/badge/Email-sandeepprajapati1202@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:sandeepprajapati1202@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-sandeepp.in-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://sandeepp.in/)

---

## 🙏 Acknowledgments

### Special Thanks

**Project Mentor:**
- **Ekshitha Namala** - For invaluable guidance and support throughout this project

**Data Sources:**
- [Zenodo Oil Spill Detection Dataset](https://zenodo.org/records/10555314) - High-quality labeled satellite imagery
- Sentinel-1 SAR, MODIS, NOAA - Satellite data providers

**Technology Partners:**
- **TensorFlow Team** - For the incredible deep learning framework
- **Streamlit** - For making web app deployment effortless
- **Supabase** - For reliable cloud database and storage

**Community:**
- **Infosys Springboard** - For the internship opportunity and resources
- **Galgotias University** - For academic support
- **Open Source Community** - For inspiration
---

*For details, sample images, or to contribute, see the repository and get in touch!*
 
<img src="https://user-images.githubusercontent.com/74038190/225813708-98b745f2-7d22-48cf-9150-083f1b00d6c9.gif" width="500">
<br><br>