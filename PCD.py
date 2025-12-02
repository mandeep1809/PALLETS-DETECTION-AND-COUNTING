import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
from PIL import Image
import numpy as np
import os

# Page setup
st.set_page_config(page_title="Pallet Detection & Counting", layout="wide")

# Load YOLO model (cached to avoid reloading)
@st.cache_resource
def load_model():
    model_path = "E:/Mandeep/360 DigiTMG/PROJECTS/Pallets Detection & Counting/Deployment/yolo11n (1).pt"  # Update this to your actual path
    return YOLO(model_path)

model = load_model()

st.title("📦 Pallet Detection and Counting")
st.markdown("Upload an image or video to detect and count pallets using a YOLO model.")

# Sidebar input option
option = st.sidebar.selectbox("Select Input Type", ("Image", "Video"))

# === IMAGE Upload ===
if option == "Image":
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Detecting pallets..."):
            img_np = np.array(image)
            result = model.predict(img_np, conf=0.5)
            res_img = result[0].plot()
            st.image(res_img, caption=f"🟩 Detected {len(result[0].boxes)} Pallets", use_container_width=True)

# === VIDEO Upload ===
elif option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        with st.spinner("Processing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                result = model.predict(frame, conf=0.5)
                annotated_frame = result[0].plot()
                stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
        os.remove(video_path)
        st.success("✅ Video processing complete.")

# Footer
st.markdown("---")
st.caption("Developed for Real-Time Pallet Detection using YOLO 🚀")
