import os
import urllib.request
import cv2
import numpy as np
import streamlit as st


@st.cache_resource
def load_cascade():
    """Ensure cascade XML file is available locally."""
    cascade_filename = "haarcascade_frontalface_default.xml"

    # Check OpenCV builtin path first
    builtin_path = os.path.join(cv2.data.haarcascades, cascade_filename)
    if os.path.exists(builtin_path):
        return cv2.CascadeClassifier(builtin_path)

    # Fallback: Download directly if missing in cloud container
    if not os.path.exists(cascade_filename):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_filename)

    return cv2.CascadeClassifier(cascade_filename)


def get_face_landmarks(img):
    """Detect face region and generate keypoints for triangulation."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = load_cascade()

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None, None

    # Get primary face bounding box
    x, y, w, h = faces[0]

    # Synthetic landmark boundary points
    landmarks = np.array(
        [
            [x, y],
            [x + w // 2, y],
            [x + w, y],
            [x + w, y + h // 2],
            [x + w, y + h],
            [x + w // 2, y + h],
            [x, y + h],
            [x, y + h // 2],
            [x + w // 4, y + h // 4],
            [x + 3 * w // 4, y + h // 4],
            [x + w // 2, y + h // 2],
            [x + w // 3, y + 3 * h // 4],
            [x + 2 * w // 3, y + 3 * h // 4],
        ],
        dtype=np.int32,
    )

    return (x, y, w, h), landmarks


def swap_faces(src_img, tgt_img):
    """Swap source face onto target body using OpenCV seamless cloning."""
    src_rect, src_pts = get_face_landmarks(src_img)
    tgt_rect, tgt_pts = get_face_landmarks(tgt_img)

    if src_rect is None or tgt_rect is None:
        return None

    tx, ty, tw, th = tgt_rect
    sx, sy, sw, sh = src_rect

    # Extract source face region and resize to target face size
    src_face = src_img[sy : sy + sh, sx : sx + sw]
    src_face_resized = cv2.resize(src_face, (tw, th))

    # Create an elliptical mask for smooth face shape
    mask = np.zeros((th, tw), dtype=np.uint8)
    center = (tw // 2, th // 2)
    axes = (tw // 2 - 2, th // 2 - 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Center coordinates where the source face will be placed on target
    center_tgt = (tx + tw // 2, ty + th // 2)

    # Perform Poisson Seamless Cloning
    output = cv2.seamlessClone(
        src_face_resized, tgt_img, mask, center_tgt, cv2.NORMAL_CLONE
    )

    return output


# Streamlit UI Setup
st.set_page_config(page_title="Simple Face Swap App", layout="centered")
st.title("Simple Face Swap App")
st.write(
    "Upload a **Face Image** and a **Body/Outfit Image** to swap the face locally."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Source Face")
    face_file = st.file_uploader(
        "Upload Face", type=["jpg", "jpeg", "png"], key="face"
    )
    if face_file:
        st.image(face_file, use_container_width=True)

with col2:
    st.subheader("2. Target Body")
    body_file = st.file_uploader(
        "Upload Body", type=["jpg", "jpeg", "png"], key="body"
    )
    if body_file:
        st.image(body_file, use_container_width=True)

if st.button("Swap Face", type="primary"):
    if face_file is not None and body_file is not None:
        # Convert uploaded bytes to OpenCV image format
        face_bytes = np.asarray(bytearray(face_file.read()), dtype=np.uint8)
        body_bytes = np.asarray(bytearray(body_file.read()), dtype=np.uint8)

        src_img = cv2.imdecode(face_bytes, cv2.IMREAD_COLOR)
        tgt_img = cv2.imdecode(body_bytes, cv2.IMREAD_COLOR)

        result = swap_faces(src_img, tgt_img)

        if result is not None:
            # Convert BGR back to RGB for Streamlit rendering
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            st.success("Face Swapped Successfully!")
            st.image(
                result_rgb, caption="Result Image", use_container_width=True
            )
        else:
            st.error(
                "Could not detect faces clearly in one or both images. Try clearer front-facing photos."
            )
    else:
        st.warning("Please upload both images before clicking Swap.")
