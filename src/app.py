import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# --- 1. PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="Fructoxis Vision", page_icon="🍎", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .result-card { padding: 24px; border-radius: 12px; margin-bottom: 24px; border: 1px solid #333; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .good-card { background: linear-gradient(145deg, rgba(30, 130, 76, 0.1), rgba(30, 130, 76, 0.2)); border-left: 5px solid #1e824c; }
    .mixed-card { background: linear-gradient(145deg, rgba(211, 84, 0, 0.1), rgba(211, 84, 0, 0.2)); border-left: 5px solid #d35400; }
    .bad-card { background: linear-gradient(145deg, rgba(192, 57, 43, 0.1), rgba(192, 57, 43, 0.2)); border-left: 5px solid #c0392b; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    </style>
""", unsafe_allow_html=True)


# --- 2. LOAD PIPELINE MODELS ---
@st.cache_resource
def load_pipeline():
    # Resolve absolute paths based on app.py location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'fructoxis_v2.keras')
    
    # Model 1: The Biochemist (Your Fine-Tuned CNN)
    fructoxis = tf.keras.models.load_model(model_path)
    return fructoxis


try:
    fructoxis_model = load_pipeline()
    models_loaded = True
except Exception as e:
    st.error(f"System Offline: Pipeline failed to initialize. {e}")
    models_loaded = False

CLASS_NAMES = [
    'Apple_Bad', 'Apple_Good', 'Apple_Mixed',
    'Banana_Bad', 'Banana_Good', 'Banana_Mixed',
    'Guava_Bad', 'Guava_Good', 'Guava_Mixed',
    'Orange_Bad', 'Orange_Good', 'Orange_Mixed'
]

# --- 3. UI HEADER ---
st.title("🍎 Fructoxis Vision Engine")
st.markdown("##### Multi-Stage Pipeline: YOLOv8 Priority Cropping + Fructoxis CNN Biochemical Estimation.")
st.divider()

# --- 4. MAIN LOGIC ---
if models_loaded:
    uploaded_file = st.file_uploader("Drop target image here (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col_img, col_res = st.columns([1, 1.2], gap="large")
        image = Image.open(uploaded_file).convert('RGB')

        with col_img:
            st.image(image, use_container_width=True, caption="Raw Feed (Stage 0)", output_format="PNG")

        with col_res:
            st.subheader("Visual Analysis Report")

            with st.spinner("Stage 1: Running YOLOv8 Spatial Detection..."):
                # --- STAGE 1: YOLO OBJECT DETECTION & PRIORITY CROPPING (ISOLATED PROCESS) ---
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                detector_script = os.path.join(BASE_DIR, "detect_fruit.py")

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as in_tmp:
                    image.save(in_tmp.name, format="PNG")
                    in_tmp_path = in_tmp.name

                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_tmp:
                    out_tmp_path = out_tmp.name

                try:
                    proc = subprocess.run(
                        [sys.executable, detector_script, in_tmp_path, out_tmp_path],
                        capture_output=True,
                        text=True
                    )
                    if proc.returncode != 0:
                        st.error(f"🚨 Spatial Detection Error: {proc.stderr or proc.stdout}")
                        st.stop()

                    with open(out_tmp_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as err:
                    st.error(f"🚨 Spatial Detection Pipeline Failure: {err}")
                    st.stop()
                finally:
                    if os.path.exists(in_tmp_path):
                        os.remove(in_tmp_path)
                    if os.path.exists(out_tmp_path):
                        os.remove(out_tmp_path)

                if data.get("status") == "no_boxes" or not data.get("target_box"):
                    st.error(
                        "🚨 **Target Lost:** YOLOv8 could not isolate a distinct object in this image. Please upload a clearer photo.")
                    st.stop()

                target_box = tuple(data["target_box"])
                yolo_guess = data.get("yolo_guess", "")
                allowed_objects = ['apple', 'banana', 'orange', 'sports ball', 'bowl']

                # --- THE GATEKEEPER ---
                if yolo_guess not in allowed_objects:
                    st.image(image.crop(target_box), width=400)
                    st.markdown(f"""
                    <div style="background-color: #2b0808; padding: 30px; border-radius: 12px; border: 2px solid #ff4b4b; text-align: center; margin-top: 20px;">
                        <h1 style="color: #ff4b4b; margin-bottom: 0px;">🛑 ACCESS DENIED</h1>
                        <h2 style="color: white; margin-top: 10px;">Object Detected: <span style="color: #ffca28; text-transform: uppercase;">{yolo_guess}</span></h2>
                        <hr style="border-color: #551515; margin: 20px 0px;">
                        <p style="color: #e0e0e0; font-size: 18px;">
                            <b>Pipeline Halted.</b> A <i>{yolo_guess}</i> is not a supported agricultural target. <br>
                            To prevent neural network hallucination, this object has been blocked from entering the Fructoxis Stage 2 Biochemical Analyzer.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()  # Halt the app completely

                # If it passes the gatekeeper, crop it and continue!
                target_image = image.crop(target_box)
                st.image(target_image, width=250, caption=f"Stage 1: YOLO Isolated Target (Detected as: {yolo_guess})")

            with st.spinner("Stage 2: Running Fructoxis Biochemical Classification..."):
                # --- STAGE 2: FRUCTOXIS CLASSIFICATION ---
                # Feed the CROPPED target to the CNN
                img_resized = target_image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = tf.expand_dims(img_array, 0)

                preds = fructoxis_model.predict(img_array)
                scores = preds[0]
                pred_idx = np.argmax(scores)

                raw_result = CLASS_NAMES[pred_idx]
                confidence = scores[pred_idx] * 100
                fruit_species, quality_state = raw_result.split('_')

                # --- IDENTIFICATION METRICS ---
                st.markdown("#### 1. Species Identification")
                st.metric(label="Primary Classification", value=fruit_species)

                st.caption(f"Network Confidence: **{confidence:.1f}%**")
                st.progress(int(confidence))
                st.write("")

                # --- QUALITY METRICS ---
                st.markdown("#### 2. Biochemical Assessment")

                if confidence < 50.0:
                    st.warning(
                        f"⚠️ **Anomaly Detected:** Neural confidence is below optimal thresholds ({confidence:.1f}%). The target may be heavily degraded or outside standard training parameters.")

                if quality_state == 'Good':
                    st.markdown(
                        f"<div class='result-card good-card'><h3>🟢 Optimal Quality</h3><p><b>State:</b> Fresh/Firm<br><b>Biochemical Estimate:</b> Low sugar concentration, intact cellular structure.</p></div>",
                        unsafe_allow_html=True)
                elif quality_state == 'Mixed':
                    st.markdown(
                        f"<div class='result-card mixed-card'><h3>🟠 Peak Ripeness</h3><p><b>State:</b> Maturing<br><b>Biochemical Estimate:</b> High sugar concentration, initial softening of tissue.</p></div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div class='result-card bad-card'><h3>🔴 Degraded Quality</h3><p><b>State:</b> Rot/Fermenting<br><b>Biochemical Estimate:</b> Advanced enzymatic breakdown. Unfit for consumption.</p></div>",
                        unsafe_allow_html=True)

            with st.expander("⚙️ View Pipeline Telemetry"):
                st.code(
                    f"YOLO Stage 1 Class: {yolo_guess}\nYOLO Box Coords: {target_box}\nFructoxis Raw Class: {raw_result}\nSoftmax Confidence: {confidence:.4f}%")
