import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import shutil

# ------------------- Init ---------------------
IGNORED_CLASSES = ["Sign", "ChqNo", "MICRCode"]
YOLO_MODEL_PATH = "runs/detect/cheque_model_v82/weights/best.pt"
OUTPUT_DIR = "out"

# Load YOLOv8 model
model = YOLO(YOLO_MODEL_PATH)

# Load TrOCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Cleanup output directory
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# ------------------- OCR Preprocessing ---------------------
def preprocess_for_trocr(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 11, 12
    )
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)  # ⬅ Convert back to 3-channel
    resized = cv2.resize(thresh_rgb, (384, 384))
    return Image.fromarray(resized)

# ------------------- Streamlit UI ---------------------
st.set_page_config(page_title="Cheque OCR", layout="wide")
st.title("🧾 Cheque Field Detection + TrOCR")

uploaded_file = st.file_uploader("Upload a cheque image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_path = "input_cheque.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(img_path, caption="📄 Uploaded Cheque", use_column_width=True)

    # Run YOLO detection
    results = model(img_path, conf=0.5)
    orig_img = cv2.imread(img_path)
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    class_names = model.names

    st.subheader("📦 Detected Fields")

    for r in results:
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            class_name = class_names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Crop
            crop = orig_img_rgb[y1:y2, x1:x2]
            pil_crop = Image.fromarray(crop)
            crop_path = os.path.join(OUTPUT_DIR, f"{class_name}_{i}.jpg")
            pil_crop.save(crop_path)

            # UI layout
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(pil_crop, caption=f"{class_name} (Conf: {conf:.2f})", use_column_width=True)

            with col2:
                if class_name in IGNORED_CLASSES:
                    st.info(f"❌ OCR skipped for `{class_name}` field.")
                else:
                    # Preprocess and resize
                    preprocessed_img = preprocess_for_trocr(pil_crop)
                    pixel_values = processor(images=preprocessed_img, return_tensors="pt").pixel_values

                    # OCR using TrOCR
                    with torch.no_grad():
                        generated_ids = trocr_model.generate(pixel_values)
                        ocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    st.success(f"**OCR Output for {class_name}:**\n{ocr_text.strip()}")

    st.success("✅ All fields processed and displayed.")
