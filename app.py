import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import shutil
import re
from word2number import w2n

# ------------------- Config ---------------------
IGNORED_CLASSES = ["Sign", "ChqNo", "MICRCode"]
NUMERIC_FIELDS = ["Amt", "DateIss"]
YOLO_MODEL_PATH = "runs/detect/cheque_model_v82/weights/best.pt"
OUTPUT_DIR = "out"

# Correction map for common OCR errors in amount in words
AMTINWORDS_CORRECTIONS = {
    "cross": "crore",
    "crass": "crore",
    "gross": "crore",
    "lacks": "lakhs",
    "Loth" : "lakhs",
    "loth": "lakhs",
    "lack": "lakh",
    "rupees": "",
    "only": "",
    "oniy": "",
    "rupes": "",
    "rupe": ""
}

# ------------------- Functions ---------------------
def preprocess_for_trocr(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(result, (384, 384))
    return Image.fromarray(resized)

def fix_amtinwords(text):
    text = text.lower()
    for wrong, right in AMTINWORDS_CORRECTIONS.items():
        text = text.replace(wrong, right)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

from datetime import datetime

def format_date(raw):
    # Extract all digit sequences
    digits = re.findall(r"\d+", raw)
    candidate = "".join(digits)
    
    # If it's exactly 8 digits, try parsing it
    if len(candidate) == 8:
        for fmt in ["%d%m%Y", "%d%m%y", "%Y%m%d"]:
            try:
                dt = datetime.strptime(candidate, fmt)
                return dt.strftime("%d/%m/%Y")
            except:
                continue

    # Try to parse from common delimiters if slashes are recognized
    try:
        dt = datetime.strptime(raw.strip(), "%d/%m/%Y")
        return dt.strftime("%d/%m/%Y")
    except:
        return raw  # fallback to original if no valid format


# ------------------- Setup Models ---------------------
model = YOLO(YOLO_MODEL_PATH)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Cleanup output dir
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# ------------------- Streamlit UI ---------------------
st.set_page_config(page_title="Cheque OCR", layout="wide")
st.title("🧾 Cheque Field Detection + TrOCR")

uploaded_file = st.file_uploader("Upload a cheque image", type=["jpg", "png", "jpeg"])

ocr_outputs = {}

if uploaded_file:
    img_path = "input_cheque.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(img_path, caption="📄 Uploaded Cheque", use_column_width=True)

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

            crop = orig_img_rgb[y1:y2, x1:x2]
            pil_crop = Image.fromarray(crop)
            crop_path = os.path.join(OUTPUT_DIR, f"{class_name}_{i}.jpg")
            pil_crop.save(crop_path)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(pil_crop, caption=f"{class_name} (Conf: {conf:.2f})", use_column_width=True)

            with col2:
                if class_name in IGNORED_CLASSES:
                    st.info(f"❌ OCR skipped for `{class_name}` field.")
                else:
                    preprocessed_img = preprocess_for_trocr(pil_crop)
                    pixel_values = processor(images=preprocessed_img, return_tensors="pt").pixel_values

                    with torch.no_grad():
                        generated_ids = trocr_model.generate(pixel_values)
                        ocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                    # Handle specific field post-processing
                    if class_name in NUMERIC_FIELDS:
                        digits_only = re.findall(r"\d+", ocr_text)
                        ocr_text = "".join(digits_only)
                        if class_name == "DateIss":
                            ocr_text = format_date(ocr_text)

                    elif class_name == "AmtinWords":
                        ocr_text = fix_amtinwords(ocr_text)

                    ocr_outputs[class_name] = ocr_text
                    st.success(f"**OCR Output for {class_name}:**\n{ocr_text}")

    # ------------------ Verification ------------------
    if "Amt" in ocr_outputs and "AmtinWords" in ocr_outputs:
        st.subheader("🔐 Amount Verification")
        amt = ocr_outputs["Amt"]
        amt_words = ocr_outputs["AmtinWords"]

        try:
            amt_from_words = w2n.word_to_num(amt_words.lower())
            amt_int = int(amt)

            if amt_from_words == amt_int:
                st.success(f"✅ Amount Verified: ₹{amt_int} == '{amt_words}'")
            else:
                st.error(f"❌ Mismatch Detected:\n\n**Amt:** ₹{amt_int}\n**AmtinWords:** {amt_words}")
        except Exception as e:
            st.warning(f"⚠️ Could not convert AmtinWords to number.\n\nError: {str(e)}")

    st.success("✅ All fields processed and displayed.")
