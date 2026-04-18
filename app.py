import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import pandas as pd

# --- Load Model ---
model = YOLO("best.pt")

# --- Class Names ---
class_names = [
    'Number_plate', 'mobile_usage', 'pillion_rider_not_wearing_helmet', 
    'rider_and_pillion_not_wearing_helmet', 'rider_not_wearing_helmet', 
    'triple_riding', 'vehicle_with_offence'
]

def format_name(name):
    return name.replace("_", " ").title()

# --- Draw Boxes (FIXED OVERLAP) ---
def draw_boxes(image, results):
    img = image.copy()
    h_img, w_img = img.shape[:2]

    if results[0].boxes is None:
        return img

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    clss = results[0].boxes.cls.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()

    used_areas = []  # track occupied label areas

    for box, cls_id, conf in zip(boxes, clss, confs):
        x1, y1, x2, y2 = box
        label = f"{format_name(class_names[cls_id])} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 🔥 Try multiple positions (top → bottom → shift)
        possible_positions = [
            (x1, y1 - 5),          # above box
            (x1, y2 + h + 5),      # below box
            (x1 + 10, y1 - 25),    # slightly right + up
            (x1 + 10, y2 + 25)     # slightly right + down
        ]

        final_x, final_y = x1, y1

        for px, py in possible_positions:
            # keep inside image
            px = max(0, min(px, w_img - w))
            py = max(h, min(py, h_img))

            overlap = False
            for ux1, uy1, ux2, uy2 in used_areas:
                if not (px + w < ux1 or px > ux2 or py < uy1 or py - h > uy2):
                    overlap = True
                    break

            if not overlap:
                final_x, final_y = px, py
                break

        # store used area
        used_areas.append((final_x, final_y - h, final_x + w, final_y))

        # draw background
        cv2.rectangle(img, (final_x, final_y - h), (final_x + w, final_y), (0, 255, 0), -1)

        # draw text
        cv2.putText(img, label, (final_x, final_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img

# --- UI CONFIG ---
st.set_page_config(page_title="Traffic Violation Dashboard", layout="wide")

with st.sidebar:
    st.title("⚙️ Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.4)

st.title("🚦 Traffic Violation Detection Dashboard")

tab1, tab2, tab3 = st.tabs(["📸 Image", "🎥 Video", "📄 About"])

# ================= IMAGE =================
with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original", use_container_width=True)

        if st.button("Detect Violations"):
            img_array = np.array(image)
            results = model.predict(img_array, conf=confidence)

            annotated = draw_boxes(img_array, results)

            detected = set()
            if results[0].boxes is not None:
                for cls_id in results[0].boxes.cls:
                    detected.add(format_name(class_names[int(cls_id)]))

            with col2:
                st.image(annotated, caption="Detection", use_container_width=True)

            st.metric("Total Violations", len(detected))

            for d in detected:
                st.write(f"🚨 {d}")

# ================= VIDEO =================
with tab2:
    uploaded_video = st.file_uploader("Upload Video", type=["mp4"])

    if uploaded_video:
        st.video(uploaded_video)

        if st.button("Run Detection"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            all_violations = set()

            # --- Video Writer ---
            output_path = "output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=confidence)
                annotated = draw_boxes(frame, results)

                if results[0].boxes is not None:
                    for cls_id in results[0].boxes.cls:
                        name = format_name(class_names[int(cls_id)])
                        all_violations.add(name)

                out.write(annotated)
                stframe.image(annotated, channels="BGR")

            cap.release()
            out.release()

            st.subheader("📊 Violations Detected")

            if all_violations:
                for v in all_violations:
                    st.write(f"🚨 {v}")
            else:
                st.write("No violations detected")

            # --- DOWNLOAD BUTTON ---
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Processed Video",
                    f,
                    file_name="processed_video.mp4"
                )

# ================= ABOUT =================
with tab3:
    st.write("""
    This project detects traffic violations using YOLOv8.
    
    ✔ Bounding box detection  
    ✔ Clean violation output  
    ✔ Video export support  
    """)

st.markdown("---")
st.write("Final Year Project - Traffic Violation Detection")