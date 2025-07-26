import streamlit as st
import cv2
import numpy as np
from PIL import Image
import datetime

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Viola-Jones Face Detection App")
st.markdown("""
Upload an image and use the sliders below to tune the face detection algorithm.
- Use `scaleFactor` to control image shrinkage.
- Use `minNeighbors` to filter out false positives.
- Choose rectangle color using the color picker.

**Click Detect Faces to begin.**
""")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Parameters
scale_factor = st.slider("scaleFactor", min_value=1.1, max_value=2.0, value=1.1, step=0.1)
min_neighbors = st.slider("minNeighbors", min_value=1, max_value=10, value=5)
rectangle_color = st.color_picker("Rectangle Color", "#00FF00")
bgr_color = tuple(int(rectangle_color.lstrip('#')[i:i+2], 16) for i in (0, 2 ,4))

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    if st.button("🔎 Detect Faces"):
        faces = face_cascade.detectMultiScale(image_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        if len(faces) == 0:
            st.warning("No faces detected. Try adjusting parameters or using a different image.")
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(image_np, (x, y), (x+w, y+h), bgr_color, 2)

            st.image(image_np, caption=f"{len(faces)} face(s) detected", use_column_width=True)

            if st.button("Save Image"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"detected_faces_{timestamp}.jpg"
                cv2.imwrite(filename, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                st.success(f"Image saved as {filename}")