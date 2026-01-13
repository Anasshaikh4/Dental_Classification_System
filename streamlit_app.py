import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import io
import os


# --- Page config & styling ---
st.set_page_config(
    page_title="Dental Classification — Inference", 
    layout="wide",
    initial_sidebar_state="expanded"
)

_CSS = """
/* Professional dark theme with clean design system */
html, body, [data-testid='stAppViewContainer'] > .main {
    height: 100vh;
    overflow: hidden;
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1419 100%);
}

.block-container {
    padding: 20px 24px 16px 24px;
    max-width: 1400px;
    margin: 0 auto;
    color: #e8edf3;
}

/* Header styling */
.header { 
    padding: 8px 0 16px 0; 
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 20px;
}
.brand { 
    color: #06b6d4; 
    font-weight: 700; 
    font-size: 24px; 
    letter-spacing: -0.5px;
}
.small-caption { 
    color: #94a3b8; 
    font-size: 14px;
    margin-top: 4px;
}

/* Column layout - equal, aligned, clean gaps */
[data-testid="stHorizontalBlock"] {
    gap: 20px !important;
}
[data-testid="stColumn"] { 
    width: 50% !important; 
    flex: 0 0 calc(50% - 10px) !important;
    align-items: flex-start;
}

/* Section headings */
.stHeading h3 {
    color: #e8edf3;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Card containers - sleek, consistent */
.card { 
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(10px);
    border-radius: 12px; 
    padding: 20px; 
    border: 1px solid rgba(148, 163, 184, 0.1);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    min-height: 520px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: flex-start;
    transition: border-color 0.2s ease;
}

.card:hover {
    border-color: rgba(148, 163, 184, 0.15);
}

/* Image display - aligned at top, consistent sizing */
.stImage { 
    display: flex !important; 
    align-items: flex-start !important; 
    justify-content: center !important;
    width: 100% !important;
    min-height: 400px;
}

.stImage img { 
    max-height: 420px !important; 
    width: auto !important; 
    max-width: 100% !important;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

/* Fullscreen image should be enlarged */
[data-testid="stFullScreenFrame"] > div[style*="position: fixed"] img,
div[data-baseweb="modal"] img,
.stImage[data-testid="stImage"] img:fullscreen {
    max-height: 90vh !important;
    max-width: 90vw !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
}

/* Modal/overlay fullscreen styling */
[data-testid="stFullScreenFrame"] [data-baseweb="modal"] {
    background: rgba(0, 0, 0, 0.95) !important;
}

[data-testid="stFullScreenFrame"] [data-baseweb="modal"] img {
    max-height: 90vh !important;
    max-width: 90vw !important;
    width: auto !important;
    height: auto !important;
}

.stImageCaption { 
    margin-top: 12px !important;
    text-align: center !important;
}

.stImageCaption p { 
    color: #94a3b8 !important;
    font-size: 13px !important;
}

/* File uploader - clean, prominent */
.stFileUploader { 
    width: 100% !important;
    margin-bottom: 16px;
}

.stFileUploader label {
    color: #e8edf3 !important;
    font-weight: 500;
    font-size: 14px;
    margin-bottom: 8px;
}

.stFileUploader section { 
    width: 100% !important;
    background: rgba(6, 182, 212, 0.05) !important;
    border: 2px dashed rgba(6, 182, 212, 0.3) !important;
    border-radius: 10px !important;
    padding: 24px !important;
    transition: all 0.2s ease;
}

.stFileUploader section:hover {
    background: rgba(6, 182, 212, 0.08) !important;
    border-color: rgba(6, 182, 212, 0.5) !important;
}

.stFileUploader .stBaseButton-secondary { 
    background: #06b6d4 !important;
    color: #0a0e1a !important;
    border: none !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
}

.stFileUploader .stBaseButton-secondary:hover {
    background: #0891b2 !important;
}

/* Download button styling */
.stDownloadButton button { 
    background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%) !important;
    color: #0a0e1a !important;
    border: none !important;
    padding: 10px 24px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 4px rgba(6, 182, 212, 0.3);
    transition: all 0.2s ease;
}

.stDownloadButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(6, 182, 212, 0.4);
}

/* Sidebar styling */
.css-1d391kg, [data-testid="stSidebar"] {
    background: rgba(10, 14, 26, 0.95) !important;
    border-right: 1px solid rgba(148, 163, 184, 0.1);
    min-width: 21rem !important;
    transform: none !important;
}

/* Disable sidebar collapse button - comprehensive selectors */
[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

button[kind="header"] {
    display: none !important;
    visibility: hidden !important;
}

[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

[data-testid="stSidebar"] button[kind="header"] {
    display: none !important;
    visibility: hidden !important;
}

/* Force sidebar to stay expanded */
section[data-testid="stSidebar"] {
    min-width: 21rem !important;
    max-width: 21rem !important;
    transform: none !important;
    left: 0 !important;
}

section[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 21rem !important;
    max-width: 21rem !important;
    transform: none !important;
    left: 0 !important;
}

/* Hide the collapse arrow/button */
[data-testid="stSidebar"] > div:first-child > button {
    display: none !important;
}

.stSidebarCollapseButton, div[data-testid="stSidebarCollapseButton"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

.css-1d391kg .stMarkdown, [data-testid="stSidebar"] .stMarkdown {
    color: #e8edf3;
}

/* Sidebar sliders */
.stSlider label {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Sidebar button */
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[kind="secondary"] {
    width: 100%;
    background: #06b6d4 !important;
    color: #0a0e1a !important;
    border: none !important;
    padding: 10px 16px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    margin-top: 8px;
}

[data-testid="stSidebar"] button:hover {
    background: #0891b2 !important;
}

/* Success/Error badges */
.stSuccess, .stError {
    padding: 6px 12px !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Spacing cleanup */
.stElementContainer { 
    margin-top: 0 !important; 
    padding-top: 0 !important; 
}

/* Remove Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


BASE_DIR = Path(__file__).resolve().parent
# Use a relative path from the application directory to the bundled model
# This points to `application/weight/best.pt` reliably across systems
MODEL_PATH = (BASE_DIR / "weight" / "yolov8s.pt").resolve()
# MODEL_PATH = Path(r"C:\Anas\Anas Working\Dental_classification\training_pipeline\runs\train7\weights\best.pt")
# Fallback model path (uncomment / adjust as needed):
# MODEL_PATH = (BASE_DIR.parent / "training_pipeline" / "yolo11n.pt").resolve()


@st.cache_resource
def load_model(path: str):
    try:
        model = YOLO(str(path))
        return model
    except Exception as e:
        return None


def infer_and_annotate(model, pil_img: Image.Image, conf=0.25, iou=0.45):
    img = np.array(pil_img.convert("RGB"))
    # Use ultralytics predict; pass numpy image as source
    results = model.predict(source=img, conf=conf, verbose=False)
    ###
    # results = model.predict(source=img, conf=conf, iou=iou, verbose=False)
    try:
        annotated = results[0].plot()
    except Exception:
        # Fallback: return original if plotting fails
        annotated = img
    annotated_pil = Image.fromarray(annotated)
    return annotated_pil, results


def resize_for_display(pil_img: Image.Image, max_w=480, max_h=540):
    """Return a resized copy that fits within max_w x max_h while preserving aspect."""
    img = pil_img.copy()
    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return img


def main():
    # Sidebar: professional, clean controls
    st.sidebar.markdown("### ⚙️ Controls")
    st.sidebar.markdown("")
    
    conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01, 
                             help="Minimum confidence score for detections")
    # iou = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01,
    #                         help="Non-maximum suppression IoU threshold")
    
    run_btn = st.sidebar.button("🚀 Run Inference", use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Model: YOLO Custom Trained")
    if MODEL_PATH.exists():
        st.sidebar.caption("✅ Model loaded successfully")
    else:
        st.sidebar.caption("❌ Model file not found")

    # Header: clean and professional
    st.markdown("<div class='header'>", unsafe_allow_html=True)
    header_col, status_col = st.columns([5, 1])
    with header_col:
        st.markdown("<span class='brand'>🦷 Dental Classification System</span>", unsafe_allow_html=True)
        st.markdown("<div class='small-caption'>AI-powered dental image analysis using YOLO object detection</div>", unsafe_allow_html=True)
    with status_col:
        if MODEL_PATH.exists():
            st.success("✓ Ready", icon="✅")
        else:
            st.error("✗ Error", icon="❌")
    st.markdown("</div>", unsafe_allow_html=True)

    # Load model once (lazy)
    model = None
    if MODEL_PATH.exists():
        with st.spinner("Loading model..."):
            model = load_model(str(MODEL_PATH))
            if model is None:
                st.error("Failed to load YOLO model. Check console logs.")

    # Two-column layout with cards
    col_in, col_out = st.columns(2, gap="medium")

    # Left column: Input preview and upload
    with col_in:
        st.markdown("### 📤 Input Image")

    # Right column: Detection result
    with col_out:
        st.markdown("### 📊 Detection Result")

    # File uploader - needs to be outside col_in to persist across reruns
    uploaded = st.sidebar.file_uploader(
        "Upload dental image", 
        type=["png", "jpg", "jpeg"], 
        key="input_uploader",
        help="Drag and drop or click to browse",
        label_visibility="collapsed"
    )
    
    # Handle no upload case
    if uploaded is None:
        placeholder = Image.new("RGB", (520, 420), color=(15,23,42))
        
        # with col_in:
        #     st.image(placeholder)
        #     st.caption("📂 No image uploaded yet")
        #     # File uploader at bottom
        #     st.file_uploader(
        #         "Upload dental image", 
        #         type=["png", "jpg", "jpeg"], 
        #         key="input_uploader_visible",
        #         help="Drag and drop or click to browse"
        #     )
        
        with col_out:
            st.image(placeholder)
            st.caption("⏳ Waiting for input image and inference...")
        
        st.info("👆 Upload an image using the file uploader above, then click 'Run Inference' in the sidebar.", icon="💡")
        return

    # Display uploaded image
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"❌ Could not open the uploaded file: {e}", icon="⚠️")
        return

    # Show input image in left column
    display_in = resize_for_display(image, max_w=520, max_h=480)
    with col_in:
        st.image(display_in)
        st.caption(f"📐 Size: {image.width} × {image.height} pixels")
        # # File uploader at bottom
        # st.file_uploader(
        #     "Upload dental image", 
        #     type=["png", "jpg", "jpeg"], 
        #     key="input_uploader_bottom",
        #     help="Drag and drop or click to browse"
        # )

    # Run inference when button clicked
    if run_btn:
        if model is None:
            st.error("❌ Model unavailable. Cannot run inference.", icon="⚠️")
            return
        
        with st.spinner("🔄 Running inference..."):
            annotated, results = infer_and_annotate(model, image, conf=conf)
            # annotated, results = infer_and_annotate(model, image, conf=conf, iou=iou)
        
        # Display output
        display_out = resize_for_display(annotated, max_w=520, max_h=480)
        with col_out:
            st.image(display_out)
            
            # Show detection summary
            try:
                boxes = results[0].boxes
                names = model.names if hasattr(model, 'names') else {}
                
                # Track class IDs with their counts
                class_counts = {}  # {class_id: count}
                total_detections = 0
                
                for b in boxes:
                    cls = int(b.cls.cpu().numpy().item()) if hasattr(b, 'cls') else None
                    if cls is not None:
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                        total_detections += 1
                
                if class_counts:
                    st.success(f"✅ {total_detections} detection(s) found", icon="🎯")
                    with st.expander("📋 Detection Details", expanded=True):
                        # Get colors from ultralytics color palette (matches bounding box colors)
                        from ultralytics.utils.plotting import colors as yolo_color_func
                        
                        for cls_id, count in sorted(class_counts.items()):
                            label = names.get(cls_id, f"Class {cls_id}")
                            # Get the actual color used by YOLO for this class
                            bgr_color = yolo_color_func(cls_id, True)  # Returns BGR tuple
                            # Convert BGR to RGB hex
                            hex_color = "#{:02x}{:02x}{:02x}".format(bgr_color[2], bgr_color[1], bgr_color[0])
                            
                            st.markdown(
                                f'<div style="display: flex; align-items: center; margin: 4px 0;">' 
                                f'<span style="display: inline-block; width: 16px; height: 16px; '
                                f'background-color: {hex_color}; border-radius: 3px; margin-right: 8px;"></span>'
                                f'<span><strong>{label}</strong>: {count}</span></div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.info("ℹ️ No detections above the confidence threshold", icon="🔍")
            except Exception:
                st.caption("Detection information unavailable")

        # Download button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                "⬇️ Download Result",
                data=buf,
                file_name="dental_detection_result.png",
                mime="image/png",
                use_container_width=True
            )
    else:
        # Show waiting state
        placeholder_wait = Image.new("RGB", (520, 420), color=(15,23,42))
        with col_out:
            st.image(placeholder_wait)
            st.info("👈 Click 'Run Inference' in the sidebar to analyze", icon="⏳")


if __name__ == "__main__":
    main()
