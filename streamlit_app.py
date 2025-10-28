# ============================================================================
# STREAMLIT_APP.PY - Premium Crystal UI Version
# ============================================================================

import streamlit as st
import numpy as np
from PIL import Image
import sys
import os
import io
import base64
import pandas as pd
from datetime import datetime
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.inference import get_detector
from utils.visualization import (create_overlay, create_confidence_heatmap,
                                 add_metrics_overlay)
from utils.preprocessing import validate_image
import config.config as cfg

# Import database functions
from utils.db import (insert_detection_data, fetch_all_detections, supabase, 
                     save_images_to_storage, fetch_detection_with_images)

# ------------------------ PAGE CONFIG --------------------------------------
st.set_page_config(
    page_title=cfg.PAGE_TITLE or "AI Oil Spill Detection",
    page_icon=cfg.PAGE_ICON or "🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# UI Enhancement: Ocean Background Video (YouTube Embed) & Custom Styling
import streamlit as st
from pathlib import Path

def inject_ocean_ui():
    """Inject YouTube video background and custom CSS styling"""

    BASE_DIR = Path(__file__).parent

    # 🎥 YouTube video embed
    yt_embed_url = "https://www.youtube.com/embed/K-D6rnKOlbk?autoplay=1&mute=1&loop=1&playlist=K-D6rnKOlbk&controls=0&showinfo=0&modestbranding=1"

    st.markdown(f"""
    <style>
        /* Fullscreen YouTube Background */
        .video-background {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            z-index: -2;
        }}

        .video-background iframe {{
            position: absolute;
            top: 50%;
            left: 50%;
            width: 120vw;
            height: 120vh;
            transform: translate(-50%, -50%);
            pointer-events: none;
            opacity: 0.7;
        }}

        /* Dark overlay for readability */
        .video-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 20, 40, 0.5);
            z-index: -1;
            pointer-events: none;
        }}
    </style>

    <div class="video-background">
        <iframe
            src="{yt_embed_url}"
            frameborder="0"
            allow="autoplay; fullscreen"
            allowfullscreen
        ></iframe>
    </div>
    <div class="video-overlay"></div>
    """, unsafe_allow_html=True)

    # 🎨 Custom CSS loading
    css_file = BASE_DIR / "styles" / "custom.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ custom.css not found in styles folder")

# Call the function
inject_ocean_ui()


# ------------------------ PREMIUM CRYSTAL BLUE STYLES ----------------------
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).parent  # Directory where your streamlit_app.py is
css_file = BASE_DIR / "styles" / "styles2.css"

with open(css_file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------ DATABASE HELPERS ---------------------------------
def init_database():
    """Initialize the detection records database in session state"""
    if 'detection_records' not in st.session_state:
        st.session_state.detection_records = []

def add_detection_record(filename, has_spill, coverage_pct, avg_confidence, max_confidence, detected_pixels):
    """Add a new detection record to the database"""
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'filename': filename,
        'result': 'Spill Detected ✅' if has_spill else 'No Spill ❌',
        'coverage_%': round(coverage_pct, 2),
        'avg_confidence': round(avg_confidence * 100, 1),
        'max_confidence': round(max_confidence * 100, 1),
        'detected_pixels': detected_pixels
    }
    st.session_state.detection_records.insert(0, record)
    
    if len(st.session_state.detection_records) > 50:
        st.session_state.detection_records = st.session_state.detection_records[:50]

def get_records_dataframe():
    """Convert records to pandas DataFrame"""
    if not st.session_state.detection_records:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.detection_records)


# ------------------------ HELPERS & MODEL ----------------------------------
@st.cache_resource
def load_model():
    """Load the model"""
    try:
        detector = get_detector(cfg.MODEL_PATH)
        return detector
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()


def process_image(detector, uploaded_file):
    """Process uploaded image and return results"""
    try:
        image = Image.open(uploaded_file).convert('RGB')

        is_valid, message = validate_image(image)
        if not is_valid:
            st.error(f"❌ Invalid image: {message}")
            return None

        with st.spinner('🔍 Analyzing image with AI...'):
            results = detector.predict(image)

        return results
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None


def image_to_bytes(img: Image.Image, fmt="PNG"):
    """Convert PIL image to bytes for downloads"""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def ensure_uint8(img):
    """Ensure image is in uint8 format"""
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img


def save_to_supabase_with_images(filename, has_spill, coverage_pct, avg_confidence, 
                                  max_confidence, detected_pixels, overlay_img, 
                                  heatmap_img, binary_mask_img):
    """
    Complete save function: metadata to database + images to storage
    """
    try:
        if supabase is None:
            print("⚠️ Supabase client not initialized")
            return False
        
        # First, upload images to storage and get URLs
        image_urls = save_images_to_storage(
            filename, overlay_img, heatmap_img, binary_mask_img
        )
        
        if image_urls is None:
            print("⚠️ Image upload failed, saving metadata without URLs")
            image_urls = {'overlay': '', 'heatmap': '', 'binary_mask': ''}
        
        # Save metadata + image URLs to database
        data = {
            'timestamp': datetime.now().isoformat(),
            'filename': str(filename),
            'has_spill': bool(has_spill),
            'coverage_percentage': float(coverage_pct),
            'avg_confidence': float(avg_confidence),
            'max_confidence': float(max_confidence),
            'detected_pixels': int(detected_pixels),
            'overlay_url': image_urls.get('overlay', ''),
            'heatmap_url': image_urls.get('heatmap', ''),
            'binary_mask_url': image_urls.get('binary_mask', '')
        }
        
        response = insert_detection_data(data, table_name="oil_detections")
        print(f"✅ Detection data saved successfully")
        return True
    
    except Exception as e:
        print(f"❌ Error in save_to_supabase_with_images: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ======================== IMAGE GALLERY FUNCTION ======================
def display_detection_image_gallery():
    """
    Display a gallery of stored detection images from Supabase with DATE FILTER ONLY
    """
    st.markdown("""
    <div class="gallery-header">
        <h2>🖼️ Detection Image Gallery</h2>
        <p>View all stored detection images from the cloud database</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        if supabase is None:
            st.info("⚠️ Supabase database not configured. Image gallery is unavailable.")
            return
        
        with st.spinner("Loading images from database..."):
            data = fetch_all_detections("oil_detections")
        
        if not data or len(data) == 0:
            st.info("🔭 No images found in the database yet. Upload and analyze images to populate the gallery.")
            return
        
        # Filter only detections that have at least one image URL
        detections_with_images = [
            d for d in data 
            if (d.get('overlay_url') and d.get('overlay_url') != '') or 
               (d.get('heatmap_url') and d.get('heatmap_url') != '') or 
               (d.get('binary_mask_url') and d.get('binary_mask_url') != '')
        ]
        
        if not detections_with_images:
            st.warning(f"⚠️ Found {len(data)} detection records, but no images are stored.")
            return
        
        st.markdown(f'<div class="gallery-success-message">✅ Found {len(detections_with_images)} detections with images</div>', unsafe_allow_html=True)
        
        # DATE FILTER ONLY
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.markdown('<h3>📅 Filter by Date</h3>', unsafe_allow_html=True)
        
        # Extract unique dates
        dates = sorted(set([d.get('timestamp', '')[:10] for d in detections_with_images if d.get('timestamp')]), reverse=True)
        
        # Create filter options - DATE ONLY
        filter_options = ["All Dates"] + dates
        
        selected_filter = st.selectbox(
            "Select a date to view detections:",
            options=filter_options,
            index=0
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Apply date filter
        filtered_detections = detections_with_images
        
        if selected_filter != "All Dates":
            filtered_detections = [d for d in detections_with_images if d.get('timestamp', '').startswith(selected_filter)]
        
        if not filtered_detections:
            st.warning(f"No detections found for date: {selected_filter}")
            return
        
        st.info(f"📊 Showing {len(filtered_detections)} of {len(detections_with_images)} detections")
        
        # Display each detection in a card
        for idx, detection in enumerate(filtered_detections):
            st.markdown(f"""
            <div class="image-gallery-card">
                <div class="gallery-header-info">
                    <div>
                        <div class="gallery-filename">📄 {detection.get('filename', 'Unknown')}</div>
                        <div class="gallery-timestamp">🕒 {detection.get('timestamp', 'Unknown')}</div>
                    </div>
                    <div>
                        <span class="gallery-status-badge {'status-detected' if detection.get('has_spill') else 'status-clean'}">
                            {'🚨 Spill Detected' if detection.get('has_spill') else '✅ Clean'}
                        </span>
                    </div>
                </div>
                <div class="gallery-metadata-box">
                    <strong>Coverage:</strong> {detection.get('coverage_percentage', 0):.2f}% | 
                    <strong>Avg Confidence:</strong> {detection.get('avg_confidence', 0):.3f} | 
                    <strong>Max Confidence:</strong> {detection.get('max_confidence', 0):.3f} | 
                    <strong>Pixels:</strong> {detection.get('detected_pixels', 0):,}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create three columns for the three image types
            col1, col2, col3 = st.columns(3)
            
            # Overlay Image
            with col1:
                overlay_url = detection.get('overlay_url', '')
                if overlay_url and overlay_url != '':
                    st.markdown("""
                    <div class="gallery-image-box">
                        <div class="gallery-image-label">Detection Overlay</div>
                    """, unsafe_allow_html=True)
                    st.image(overlay_url, use_column_width=True)
                    st.markdown(f'<a href="{overlay_url}" target="_blank" class="image-url-link">🔗 Open Full Image</a>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="no-image-box">⚠️ Overlay image not available</div>', unsafe_allow_html=True)
            
            # Heatmap Image
            with col2:
                heatmap_url = detection.get('heatmap_url', '')
                if heatmap_url and heatmap_url != '':
                    st.markdown("""
                    <div class="gallery-image-box">
                        <div class="gallery-image-label">Confidence Heatmap</div>
                    """, unsafe_allow_html=True)
                    st.image(heatmap_url, use_column_width=True)
                    st.markdown(f'<a href="{heatmap_url}" target="_blank" class="image-url-link">🔗 Open Full Image</a>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="no-image-box">⚠️ Heatmap image not available</div>', unsafe_allow_html=True)
            
            # Binary Mask Image
            with col3:
                binary_url = detection.get('binary_mask_url', '')
                if binary_url and binary_url != '':
                    st.markdown("""
                    <div class="gallery-image-box">
                        <div class="gallery-image-label">Binary Mask</div>
                    """, unsafe_allow_html=True)
                    st.image(binary_url, use_column_width=True)
                    st.markdown(f'<a href="{binary_url}" target="_blank" class="image-url-link">🔗 Open Full Image</a>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="no-image-box">⚠️ Binary mask image not available</div>', unsafe_allow_html=True)
            
            st.markdown('<div style="margin: 30px 0;"></div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error loading image gallery: {str(e)}")
        print(f"Gallery error: {e}")
        import traceback
        traceback.print_exc()

# ==================== MAIN UI -----------------------------------------
def main():
    # Initialize database
    init_database()
    
    # Initialize session state
    if 'total_processed' not in st.session_state:
        st.session_state.total_processed = 0
    if 'total_detections' not in st.session_state:
        st.session_state.total_detections = 0

    # ==================== HERO SECTION ====================
    st.markdown("""
    <div class="hero-section">
        <span class="emoji-icon">🌊🌊</span>
        <h1>HydroVexel⚡️</h1>
        <h4>AI-Powered Oil Spill Detection System 🤖 </h4>
                <br>
        <p style="font-family:Poppins,sans-serif;font-style:italic;font-weight:300;font-size:1.2rem;color:#ffffff;"> <b> ~ AI</b> Eyes Safeguarding Our <b>Precious Oceans</b>✨️</p>
        <p class="subtitle">
            Our system leverages cutting-edge Deep Learning and AI technologies to detect and analyze oil spills from satellite and aerial imagery with high speed and accuracy. Designed for environmental monitoring agencies, researchers, and response teams, it transforms raw imagery into actionable insights, helping protect marine ecosystems and coastal communities.
        </p>
        <p class="author"><a href="https://www.linkedin.com/in/simplysandeepp/" target="_blank" style="color: inherit; text-decoration: none;"> Developed by Sandeep Prajapati ♥️</a></p>

    </div>
    """, unsafe_allow_html=True)

    # ==================== ABOUT SECTION ====================
    st.markdown("""
    <div class="cards-container">
        <div class="info-card">
            <span class="icon">🌊</span>
            <h3>What is an Oil Spill?</h3>
            <p>An oil spill is the release of liquid petroleum hydrocarbons into the environment, especially marine areas. These incidents can devastate marine ecosystems, kill wildlife, contaminate water sources, and cause long-lasting environmental damage that affects coastal communities and economies.</p>
        </div>
        <div class="info-card">
            <span class="icon">🛰️</span>
            <h3>Why Early Detection Matters</h3>
            <p>Early detection is critical for effective response. The faster we identify oil spills, the quicker cleanup crews can be deployed, containment strategies can be implemented, and environmental damage can be minimized. Time is the most crucial factor in spill response operations.</p>
        </div>
        <div class="info-card">
            <span class="icon">⚙️</span>
            <h3>How AI Helps</h3>
            <p>Deep learning models analyze satellite and aerial imagery at scale, identifying potential oil spills with high accuracy. AI can process thousands of images in minutes, detecting patterns invisible to the human eye, and enabling rapid response to environmental threats worldwide.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==================== UPLOAD SECTION ====================
    st.markdown('<h2 class="section-title">📤 Upload & Analyze Imagery</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Upload satellite or aerial imagery to detect potential oil spills using our AI model</p>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="For best results, use high-resolution satellite or aerial imagery"
    )

    # Controls in columns
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=cfg.CONFIDENCE_THRESHOLD,
            step=0.01,
            help="Minimum confidence level for detection"
        )
    with col2:
        overlay_alpha = st.slider(
            "Overlay Transparency",
            min_value=0.0,
            max_value=1.0,
            value=cfg.OVERLAY_ALPHA,
            step=0.05,
            help="Transparency of the detection overlay"
    )

    # Action buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        detect_button = st.button("🔍 Detect Oil Spill", type="primary", use_container_width=True)
    with col_btn2:
        clear_button = st.button("♻️ Clear Results", use_container_width=True)

    # Handle clear button
    if clear_button:
        st.session_state.total_processed = 0
        st.session_state.total_detections = 0
        st.rerun()

    # Update config values
    cfg.CONFIDENCE_THRESHOLD = confidence_threshold
    cfg.OVERLAY_ALPHA = overlay_alpha

    # ==================== RESULTS SECTION ====================
    if uploaded_file is not None and detect_button:
        detector = load_model()
        results = process_image(detector, uploaded_file)

        if results is None:
            st.warning("⚠️ Image could not be processed. Please upload a valid satellite or aerial image.")
        else:
            # Update session stats
            st.session_state.total_processed += 1
            if results['metrics']['has_spill']:
                st.session_state.total_detections += 1

            # Add record to local session database
            add_detection_record(
                filename=uploaded_file.name,
                has_spill=results['metrics']['has_spill'],
                coverage_pct=results['metrics']['coverage_percentage'],
                avg_confidence=results['metrics']['avg_confidence'],
                max_confidence=results['metrics']['max_confidence'],
                detected_pixels=results['metrics']['detected_pixels']
            )

            # Ensure all images are uint8 for proper color display
            original_img = ensure_uint8(results['original_image'])
            binary_mask = ensure_uint8(results['binary_mask'])
            
            # Create visualizations with proper color handling
            overlay = create_overlay(
                original_img,
                binary_mask,
                alpha=overlay_alpha
            )
            overlay = ensure_uint8(overlay)
            
            heatmap = create_confidence_heatmap(results['confidence_map'], original_img)
            heatmap = ensure_uint8(heatmap)

            # Convert to PIL Images for storage upload
            overlay_pil = Image.fromarray(overlay)
            heatmap_pil = Image.fromarray(heatmap)
            binary_mask_pil = Image.fromarray(binary_mask)

            # Save to Supabase database with images
            save_success = False
            if overlay is not None and heatmap is not None and binary_mask is not None:
                save_success = save_to_supabase_with_images(
                    filename=uploaded_file.name,
                    has_spill=bool(results['metrics']['has_spill']),
                    coverage_pct=float(results['metrics']['coverage_percentage']),
                    avg_confidence=float(results['metrics']['avg_confidence']),
                    max_confidence=float(results['metrics']['max_confidence']),
                    detected_pixels=int(results['metrics']['detected_pixels']),
                    overlay_img=overlay_pil,
                    heatmap_img=heatmap_pil,
                    binary_mask_img=binary_mask_pil
                )
                
                if save_success:
                    st.success("✅ Detection data and images saved to cloud database!")
                else:
                    st.warning("⚠️ Images processed but cloud upload had issues. Check logs.")

            # Detection status with high-contrast badge
            if results['metrics']['has_spill']:
                st.markdown(f"""
                <div style="text-align: center; margin: 30px 0;">
                    <div class="detection-status status-detected">
                        🚨 OIL SPILL DETECTED
                    </div>
                    <p style="font-size: 1.4rem; color: #ffffff; font-weight: 700; margin-top: 15px;">
                        Coverage: {results['metrics']['coverage_percentage']:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; margin: 30px 0;">
                    <div class="detection-status status-clean">
                        ✅ NO OIL SPILL DETECTED
                    </div>
                    <p style="font-size: 1.4rem; color: #ffffff; font-weight: 700; margin-top: 15px;">
                        Area is clean
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Results in properly aligned columns
            col1, col2, col3 = st.columns(3, gap="medium")
            
            # Column 1: Detection Overlay
            with col1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<h3 style="font-family: Poppins, sans-serif; color: #ffffff; margin-bottom: 20px; font-size: 1.5rem; font-weight: 900; text-align: center; padding: 16px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border-radius: 12px; box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);">Detection Overlay</h3>', unsafe_allow_html=True)
                st.image(overlay, use_column_width=True, channels="RGB")
                st.download_button(
                    "📥 Download Overlay",
                    data=image_to_bytes(overlay_pil),
                    file_name='oil_spill_overlay.png',
                    mime='image/png',
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # Column 2: Confidence Heatmap
            with col2:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<h3 style="font-family: Poppins, sans-serif; color: #ffffff; margin-bottom: 20px; font-size: 1.5rem; font-weight: 900; text-align: center; padding: 16px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border-radius: 12px; box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);">Confidence Heatmap</h3>', unsafe_allow_html=True)
                st.image(heatmap, use_column_width=True, channels="RGB")
                st.download_button(
                    "📥 Download Heatmap",
                    data=image_to_bytes(heatmap_pil),
                    file_name='oil_spill_heatmap.png',
                    mime='image/png',
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # Column 3: Metrics
            with col3:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<h3 style="font-family: Poppins, sans-serif; color: #ffffff; margin-bottom: 20px; font-size: 1.5rem; font-weight: 900; text-align: center; padding: 16px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border-radius: 12px; box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);">Detection Metrics</h3>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['coverage_percentage']:.2f}%</div>
                    <div class="metric-label">Coverage Area</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['avg_confidence']:.1%}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['max_confidence']:.1%}</div>
                    <div class="metric-label">Max Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['detected_pixels']:,}</div>
                    <div class="metric-label">Detected Pixels</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

            # Additional details in tabs
            st.markdown("<br>", unsafe_allow_html=True)
            tab1, tab2, tab3 = st.tabs(["📊 Binary Mask", "📋 Raw JSON Data", "📈 Analysis Summary"])

            with tab1:
                st.markdown('<div class="tab-content-box">', unsafe_allow_html=True)
                st.image(
                    binary_mask,
                    caption='Binary segmentation mask (white = oil spill detected)',
                    use_column_width=True,
                    clamp=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.markdown('<div class="tab-content-box">', unsafe_allow_html=True)
                st.json({
                    'detection_status': 'Spill Detected' if results['metrics']['has_spill'] else 'No Spill',
                    'coverage_percentage': float(results['metrics']['coverage_percentage']),
                    'detected_pixels': int(results['metrics']['detected_pixels']),
                    'total_pixels': int(results['metrics']['total_pixels']),
                    'average_confidence': float(results['metrics']['avg_confidence']),
                    'maximum_confidence': float(results['metrics']['max_confidence']),
                    'threshold_used': float(confidence_threshold),
                    'overlay_alpha': float(overlay_alpha)
                })
                st.markdown('</div>', unsafe_allow_html=True)

            with tab3:
                st.markdown('<div class="tab-content-box"><div class="analysis-summary">', unsafe_allow_html=True)
                
                st.markdown(f"""
                ### Analysis Summary
                
                **Detection Result:** {'✅ Oil Spill Detected' if results['metrics']['has_spill'] else '❌ No Oil Spill Detected'}
                
                **Coverage Analysis:**
                - Total area analyzed: <strong>{results['metrics']['total_pixels']:,} pixels</strong>
                - Contaminated area: <strong>{results['metrics']['detected_pixels']:,} pixels</strong>
                - Coverage percentage: <strong>{results['metrics']['coverage_percentage']:.2f}%</strong>
                
                **Confidence Metrics:**
                - Average confidence: <strong>{results['metrics']['avg_confidence']:.1%}</strong>
                - Maximum confidence: <strong>{results['metrics']['max_confidence']:.1%}</strong>
                - Detection threshold: <strong>{confidence_threshold:.1%}</strong>
                
                **Recommendations:**
                {('<strong>- Immediate response required for cleanup operations</strong>' if results['metrics']['coverage_percentage'] > 5 else '<strong>- Monitor the area for potential expansion</strong>') if results['metrics']['has_spill'] else '<strong>- Continue routine monitoring</strong>'}
                """, unsafe_allow_html=True)
                
                st.markdown('</div></div>', unsafe_allow_html=True)

    # ==================== STATISTICS BANNER ====================
    st.markdown(f"""
    <div class="stats-banner">
        <div class="stat">
            <div class="stat-number">{st.session_state.total_processed}</div>
            <div class="stat-label">Images Analyzed</div>
        </div>
        <div class="stat">
            <div class="stat-number">{st.session_state.total_detections}</div>
            <div class="stat-label">Spills Detected</div>
        </div>
        <div class="stat">
            <div class="stat-number">{(st.session_state.total_detections / max(st.session_state.total_processed, 1) * 100):.1f}%</div>
            <div class="stat-label">Detection Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # ==================== LIVE DATABASE SECTION ====================
    if st.session_state.detection_records:
        
        st.markdown("""
        <div class="database-header">
            <h2 class="section-title">📊 Live Detection Database (Session)</h2>
        </div>
        """, unsafe_allow_html=True)
        
        df = get_records_dataframe()
        total_records = len(df)
        spills_found = len(df[df['result'].str.contains('✅')])
        avg_coverage = df['coverage_%'].mean() if not df.empty else 0
        
        st.markdown(f"""
        <div class="db-stats">
            <div class="db-stat-box">
                <div class="db-stat-value">{total_records}</div>
                <div class="db-stat-label">Total Records</div>
            </div>
            <div class="db-stat-box">
                <div class="db-stat-value">{spills_found}</div>
                <div class="db-stat-label">Spills Found</div>
            </div>
            <div class="db-stat-box">
                <div class="db-stat-value">{avg_coverage:.1f}%</div>
                <div class="db-stat-label">Avg Coverage</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                "filename": st.column_config.TextColumn("Image File", width="medium"),
                "result": st.column_config.TextColumn("Result", width="small"),
                "coverage_%": st.column_config.NumberColumn("Coverage %", format="%.2f"),
                "avg_confidence": st.column_config.NumberColumn("Avg Conf %", format="%.1f"),
                "max_confidence": st.column_config.NumberColumn("Max Conf %", format="%.1f"),
                "detected_pixels": st.column_config.NumberColumn("Pixels", format="%d")
            }
        )
        
        col_export1, col_export2, col_export3 = st.columns([1, 1, 2])
        with col_export1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export CSV",
                data=csv,
                file_name=f'detection_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                use_container_width=True
            )
        with col_export2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Export JSON",
                data=json_data,
                file_name=f'detection_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json',
                use_container_width=True
            )
        with col_export3:
            if st.button("🗑️ Clear All Records", use_container_width=True):
                st.session_state.detection_records = []
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="database-section">
            <h2 class="section-title">📊 Live Detection Database (Session)</h2>
            <p style="text-align: center; color: #4A5568; padding: 40px 0; font-size: 1.1rem;">
                No records yet. Upload and analyze images to see detection history here.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ==================== IMAGE GALLERY SECTION (NEW!) ====================
    st.markdown('<div style="margin-top: 60px;"></div>', unsafe_allow_html=True)
    display_detection_image_gallery()

    # ==================== PREVIOUS DETECTIONS FROM SUPABASE ====================
    st.markdown('<div style="margin-top: 60px;"></div>', unsafe_allow_html=True)    
    st.markdown("""
    <div class="database-header">
        <h2 class="section-title">📊 Previous Detections (Database Records)</h2>
        <p style="color: #1E1E1E; font-size: 1.15rem; margin-top: 10px; font-weight: 600;">
            All detection metadata stored in the cloud database
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        if supabase is None:
            st.info("⚠️ Supabase database not configured. Only session data is available.")
        else:
            with st.spinner("Loading previous detections from database..."):
                data = fetch_all_detections("oil_detections")
            
            if data and len(data) > 0:
                df_supabase = pd.DataFrame(data)
                
                if 'timestamp' in df_supabase.columns:
                    df_supabase['timestamp'] = pd.to_datetime(df_supabase['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                if 'has_spill' in df_supabase.columns:
                    df_supabase['result'] = df_supabase['has_spill'].apply(lambda x: 'Spill Detected ✅' if x else 'No Spill ❌')
                
                total_db_records = len(df_supabase)
                spills_in_db = df_supabase['has_spill'].sum() if 'has_spill' in df_supabase.columns else 0
                avg_coverage_db = df_supabase['coverage_percentage'].mean() if 'coverage_percentage' in df_supabase.columns else 0
                
                st.markdown(f"""
                <div class="db-stats">
                    <div class="db-stat-box">
                        <div class="db-stat-value">{total_db_records}</div>
                        <div class="db-stat-label">Total Detections</div>
                    </div>
                    <div class="db-stat-box">
                        <div class="db-stat-value">{spills_in_db}</div>
                        <div class="db-stat-label">Spills Detected</div>
                    </div>
                    <div class="db-stat-box">
                        <div class="db-stat-value">{avg_coverage_db:.1f}%</div>
                        <div class="db-stat-label">Avg Coverage</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                display_columns = []
                column_config = {}
                
                if 'timestamp' in df_supabase.columns:
                    display_columns.append('timestamp')
                    column_config['timestamp'] = st.column_config.TextColumn("Timestamp", width="medium")
                
                if 'filename' in df_supabase.columns:
                    display_columns.append('filename')
                    column_config['filename'] = st.column_config.TextColumn("Image File", width="medium")
                
                if 'result' in df_supabase.columns:
                    display_columns.append('result')
                    column_config['result'] = st.column_config.TextColumn("Result", width="small")
                
                if 'coverage_percentage' in df_supabase.columns:
                    display_columns.append('coverage_percentage')
                    column_config['coverage_percentage'] = st.column_config.NumberColumn("Coverage %", format="%.2f")
                
                if 'avg_confidence' in df_supabase.columns:
                    display_columns.append('avg_confidence')
                    column_config['avg_confidence'] = st.column_config.NumberColumn("Avg Confidence", format="%.3f")
                
                if 'max_confidence' in df_supabase.columns:
                    display_columns.append('max_confidence')
                    column_config['max_confidence'] = st.column_config.NumberColumn("Max Confidence", format="%.3f")
                
                if 'detected_pixels' in df_supabase.columns:
                    display_columns.append('detected_pixels')
                    column_config['detected_pixels'] = st.column_config.NumberColumn("Detected Pixels", format="%d")
                
                st.dataframe(
                    df_supabase[display_columns] if display_columns else df_supabase,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config
                )
                
                col_db1, col_db2 = st.columns(2)
                with col_db1:
                    csv_db = df_supabase.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Export All Detections (CSV)",
                        data=csv_db,
                        file_name=f'supabase_detections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                with col_db2:
                    json_db = df_supabase.to_json(orient='records', indent=2)
                    st.download_button(
                        "📥 Export All Detections (JSON)",
                        data=json_db,
                        file_name=f'supabase_detections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                        mime='application/json',
                        use_container_width=True
                    )
            else:
                st.info("🔭 No previous detections found in the database. Upload and analyze images to populate the database.")
    
    except Exception as e:
        st.error(f"❌ Error loading previous detections: {str(e)}")
        print(f"Database error: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # ==================== FOOTER ====================
    st.markdown("""
    <div class="footer">
        <p style="margin-bottom: 10px;">🌊 <strong>HydroVexel</strong> - Protecting Our Oceans with AI</p>
        <p class="author"><a href="https://www.linkedin.com/in/simplysandeepp/" target="_blank" style="color: inherit; text-decoration: none;"> Developed by Sandeep Prajapati ♥️</a></p>
        <p style="margin-top: 10px; font-size: 0.95rem; opacity: 0.9;">
            Powered by Deep Learning • Streamlit • Supabase
        </p>
        <p style="margin-top: 15px; font-size: 0.85rem; opacity: 0.8;">
            © 2025 HydroVexel. All rights reserved. | For environmental monitoring and research purposes.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
