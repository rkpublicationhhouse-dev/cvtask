import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. UI Setup
st.set_page_config(page_title="CV Tasks Demo", layout="wide")

st.markdown("""
    <style>
    .stMarkdown p, .stText { font-size: 20px !important; }
    h1 { font-size: 48px !important; color: #1E88E5; }
    h2, h3 { font-size: 32px !important; }
    .stSelectbox label { font-size: 18px !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("Computer Vision Tasks Demonstration")
st.info("**Tip:** For the most accurate results, please upload simple images of cats or dogs. Avoid overlapping objects or cluttered backgrounds.")

# --- CLEAR LOGIC: Detect Mode Change ---
# We store the 'mode' in session state to compare it on every rerun
if 'previous_mode' not in st.session_state:
    st.session_state['previous_mode'] = "Single Object"
    st.session_state['uploader_key'] = 0

# --- SIDEBAR ---
st.sidebar.header("Choose the options")

mode = st.sidebar.selectbox(
    "1. Select Mode",
    ("Single Object", "Multiple Objects")
)

# If the mode changed, increment the key to reset the file uploader
if mode != st.session_state['previous_mode']:
    st.session_state['previous_mode'] = mode
    st.session_state['uploader_key'] += 1
    st.rerun() # Refresh the app to clear the uploader immediately

if mode == "Single Object":
    task = st.sidebar.selectbox("2. Select Task", ("Classification", "Classification + Localisation"))
else:
    task = st.sidebar.selectbox("2. Select Task", ("Object Detection", "Instance Segmentation"))

st.sidebar.divider()
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.0, 1.0, 0.40)

# --- MODEL LOGIC ---
@st.cache_resource
def get_model(task_name):
    if task_name == "Classification":
        return YOLO("yolov8n-cls.pt")
    elif task_name == "Instance Segmentation":
        return YOLO("yolov8n-seg.pt")
    else:
        return YOLO("yolov8n.pt")

model = get_model(task)

# --- IMAGE UPLOAD (Using dynamic key to force reset) ---
# The key changes whenever 'mode' changes, clearing the uploaded file
uploaded_file = st.file_uploader(
    "📤 Upload an image to analyze...", 
    type=["jpg", "png", "jpeg"],
    key=f"uploader_{st.session_state['uploader_key']}"
)

if uploaded_file:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader(f"AI: {task}")
        
        if task == "Classification + Localisation":
            results = model(img, conf=conf_threshold, max_det=1)
        else:
            results = model(img, conf=conf_threshold)
        
        res_plotted = results[0].plot(line_width=4, font_size=20)
        st.image(res_plotted, use_container_width=True, channels="BGR")
        
        if task == "Classification":
            top1 = results[0].probs.top1
            label = results[0].names[top1]
            st.success(f"**Identified:** {label.capitalize()}")
        else:
            count = len(results[0].boxes)
            st.info(f"**Found:** {count} object(s)")

