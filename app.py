import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import numpy as np
import torch
import os
import io

# Set page configuration
st.set_page_config(page_title="Blood Cell Detection", layout="wide", page_icon="🔬")

# Custom CSS for a modern, clean UI
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    h1 {
        font-weight: 700;
        color: #1e40af;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
    }
    h2, h3 {
        color: #1e40af;
        font-weight: 600;
    }
    .stat-card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    .sidebar .css-1d391kg {
        background-color: #f1f5f9;
    }
    .stButton > button, .stDownloadButton > button {
        background-color: #3b82f6;
        color: white;
        font-weight: 500;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .cell-count {
        font-size: 1.5rem;
        font-weight: 700;
    }
    .cell-type {
        font-size: 1rem;
        color: #64748b;
    }
    .results-container {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
    }
    .header-icon {
        font-size: 2.5rem;
        margin-right: 1rem;
        color: #1e40af;
    }
    </style>
""", unsafe_allow_html=True)

# Load fine-tuned model
@st.cache_resource
def load_model():
    model = YOLO("models/finetuned_yolov10s.pt")
    return model

model = load_model()
class_names = ['WBC', 'RBC', 'Platelets']

# UI Header
st.markdown("""
    <div class="header-container">
        <div class="header-icon">🔬</div>
        <div>
            <h1>Blood Cell Analysis Dashboard</h1>
            <p>Detect and count white blood cells, red blood cells, and platelets in blood smear images</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar with settings
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.01)
    
    st.markdown("---")
    
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        This dashboard uses a fine-tuned YOLOv10 model to detect and count blood cells in microscopic images.
        
        **Cell Types:**
        - WBC: White Blood Cells
        - RBC: Red Blood Cells
        - Platelets
        """)
    
    st.markdown("---")
    uploaded_images = st.file_uploader("Upload Blood Smear Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Main body
if uploaded_images:
    for img_file in uploaded_images:
        st.markdown(f"## Analysis: {img_file.name}")
        image = Image.open(img_file).convert("RGB")
        
        # Get predictions
        results = model.predict(image)
        result = results[0]
        boxes = result.boxes
        
        # Layout in columns for image display
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='results-container'>", unsafe_allow_html=True)
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Annotated image
        annotated_img = result.plot()
        with col2:
            st.markdown("<div class='results-container'>", unsafe_allow_html=True)
            st.subheader("Detected Cells")
            st.image(annotated_img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Prediction summary
        if boxes and boxes.cls is not None:
            classes = [class_names[int(cls)] for cls in boxes.cls.cpu().numpy()]
            confidences = [round(float(c), 3) for c in boxes.conf.cpu().numpy()]
            df = pd.DataFrame({"Class": classes, "Confidence": confidences})
            
            # Filter based on confidence threshold
            filtered_df = df[df['Confidence'] >= threshold]
            
            # Class-wise stats
            summary = filtered_df.groupby('Class').agg(
                Count=('Class', 'count'),
                Avg_Confidence=('Confidence', 'mean')
            ).reset_index()
            
            # Summary Cards
            st.markdown("## Cell Count Summary")
            
            # Create cards for each cell type
            cols = st.columns(3)
            cell_types = ['WBC', 'RBC', 'Platelets']
            cell_icons = ['🟡', '🔴', '🟣']
            
            for i, cell_type in enumerate(cell_types):
                with cols[i]:
                    st.markdown(f"<div class='stat-card'>", unsafe_allow_html=True)
                    cell_count = summary[summary['Class'] == cell_type]['Count'].values[0] if not summary[summary['Class'] == cell_type].empty else 0
                    st.markdown(f"<div class='cell-type'>{cell_icons[i]} {cell_type}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='cell-count'>{cell_count}</div>", unsafe_allow_html=True)
                    if not summary[summary['Class'] == cell_type].empty:
                        avg_conf = summary[summary['Class'] == cell_type]['Avg_Confidence'].values[0]
                        st.markdown(f"<div>Avg. Confidence: {avg_conf:.3f}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Detailed results
            st.markdown("## Detailed Results")
            tab1, tab2 = st.tabs(["Filtered Detections", "Summary Statistics"])
            
            with tab1:
                st.dataframe(filtered_df, use_container_width=True)
            
            with tab2:
                st.dataframe(summary, use_container_width=True)
            
            # Download CSV option
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Detection Report",
                data=csv,
                file_name=f'{img_file.name}_detection_report.csv',
                mime='text/csv',
                use_container_width=True
            )
        else:
            st.warning("No cells detected. Try adjusting the confidence threshold.")
else:
    # Empty state
    st.markdown("<div class='results-container' style='text-align: center; padding: 3rem;'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload Images to Start")
    st.markdown("Upload one or more blood smear images from the sidebar to begin detection and analysis.")
    st.markdown("</div>", unsafe_allow_html=True)