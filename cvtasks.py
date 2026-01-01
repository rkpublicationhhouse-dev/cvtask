import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. UI Setup: Force wide layout and high-visibility text
st.set_page_config(page_title="CV Tasks Demo", layout="wide")

st.markdown("""
    <style>
    /* Make standard text 20px for readability */
    .stMarkdown p, .stText { font-size: 20px !important; }
    /* Make headers stand out */
    h1 { font-size: 48px !important; color: #1E88E5; }
    h2, h3 { font-size: 32px !important; }
    /* Style the sidebar labels */
    .stSelectbox label { font-size: 18px !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("Computer Vision Tasks Demonstration")
st.info("** Tip:** For the most accurate results, please upload simple images of cats or dogs. Avoid overlapping objects or cluttered backgrounds.")
# --- SIDEBAR: HIERARCHICAL DROPDOWNS ---
st.sidebar.header("Navigation")

# Dropdown 1: Mode
mode = st.sidebar.selectbox(
    "1. Select Mode",
    ("Single Object", "Multiple Objects")
)

# Dropdown 2: Task (Conditional based on Dropdown 1)
if mode == "Single Object":
    task = st.sidebar.selectbox(
        "2. Select Task",
        ("Classification", "Classification + Localisation")
    )
else:
    task = st.sidebar.selectbox(
        "2. Select Task",
        ("Object Detection", "Instance Segmentation")
    )

st.sidebar.divider()

# Confidence Slider (Helps solve the 'bird as bottle' issue)
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.0, 1.0, 0.40)

# --- MODEL LOGIC (Strictly Nano) ---
@st.cache_resource
def get_model(task_name):
    if task_name == "Classification":
        return YOLO("yolov8n-cls.pt")
    elif task_name == "Instance Segmentation":
        return YOLO("yolov8n-seg.pt")
    else:
        # Both Detection and Localisation use the base model
        return YOLO("yolov8n.pt")

model = get_model(task)

# --- IMAGE UPLOAD (Single Option) ---
uploaded_file = st.file_uploader("📤 Upload an image to analyze...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        # 'use_container_width' resizes image to fit the column exactly
        st.image(img, use_container_width=True)

    with col2:
        st.subheader(f"AI: {task}")
        
        # Inference Logic
        # For 'Localisation', we limit the AI to just one detection (max_det=1)
        if task == "Classification + Localisation":
            results = model(img, conf=conf_threshold, max_det=1)
        else:
            results = model(img, conf=conf_threshold)
        
        # VISIBILITY FIX: 'font_size' and 'line_width' make labels like 'parrot' readable
        res_plotted = results[0].plot(line_width=4, font_size=20)
        st.image(res_plotted, use_container_width=True, channels="BGR")
        
        # Result Details
        if task == "Classification":
            top1 = results[0].probs.top1
            label = results[0].names[top1]
            st.success(f"**Identified:** {label.capitalize()}")
        else:
            count = len(results[0].boxes)
            st.info(f"**Found:** {count} object(s)")

