import streamlit as st
import requests
import base64
from PIL import Image
import io
from datetime import datetime

st.set_page_config(page_title="X-ray Quality Control Classifier", page_icon="🔬", layout="centered")

API_KEY = st.secrets["ROBOFLOW_API_KEY"]
WORKSPACE = "zara-ashraf"
WORKFLOW_ID = "xray-qc-classifier-vxray-qc-classifier-qq9jj-1-vit-base-patch16-224-in21k-t1-logic",
CLASS_INFO = {
    "Good_Quality": "The X-ray meets quality standards - no visible defects.",
    "Blur": "The image shows motion blur or focus issues.",
    "Exposure_Error": "The image is over-exposed or under-exposed.",
    "Foreign_Artifact": "An unexpected object (metal, jewelry, hardware) is visible in the image.",
}

if "history" not in st.session_state:
    st.session_state.history = []
if "selected_image_path" not in st.session_state:
    st.session_state.selected_image_path = None

with st.sidebar:
    st.header("About this project")
    st.write("This tool automatically checks radiograph images for common quality-control issues using a custom-trained Vision Transformer (ViT) model.")
    st.subheader("Detected Classes")
    for cls, desc in CLASS_INFO.items():
        st.markdown("**" + cls.replace('_', ' ') + "**")
        st.caption(desc)
    st.markdown("---")
    st.caption("Developed by Zara Ashraf")

st.markdown("<h1 style='text-align: center;'>X-ray Quality Control Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-powered radiograph quality assessment tool</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 0.9em;'>by Zara Ashraf</p>", unsafe_allow_html=True)
st.markdown("---")

st.info("This is a demonstration prototype trained on a limited, self-curated dataset. It's a proof-of-concept that shows the approach works, accuracy will continue to improve as the dataset grows and the model is retrained.")

with st.expander("How it works"):
    st.write("This classifier uses a Vision Transformer (ViT) model trained on a custom-built dataset of chest and body-part X-rays. The model was trained to recognize four quality categories: Good Quality, Blur, Exposure Error, and Foreign Artifact. Images are processed through a Roboflow-hosted inference workflow, which returns a confidence score for each category.")

st.subheader("Try a sample image")
sample_files = [
    ".streamlit/sample 1.jpg",
    ".streamlit/sample 2.png",
    ".streamlit/sample 3.png",
    ".streamlit/sample 4.jpeg",
]
sample_cols = st.columns(4)
for i, col in enumerate(sample_cols):
    with col:
        if st.button("Sample " + str(i + 1), use_container_width=True):
            st.session_state.selected_image_path = sample_files[i]

st.markdown("---")

uploaded_file = st.file_uploader("Or upload your own X-ray image", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "jfif"])

image_source = None
image_name = None

if uploaded_file is not None:
    image_source = Image.open(uploaded_file).convert("RGB")
    image_name = uploaded_file.name
    st.session_state.selected_image_path = None
elif st.session_state.selected_image_path is not None:
    try:
        image_source = Image.open(st.session_state.selected_image_path).convert("RGB")
        image_name = st.session_state.selected_image_path
    except FileNotFoundError:
        st.warning("Sample image not found.")

if image_source is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(image_source, caption="Selected X-ray", use_container_width=True)

    buffered = io.BytesIO()
    image_source.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    with st.spinner("Running quality checks on your image..."):
        try:
            url = "https://serverless.roboflow.com/infer/workflows/" + WORKSPACE + "/" + WORKFLOW_ID
            payload = {
                "api_key": API_KEY,
                "inputs": {
                    "image": {"type": "base64", "value": img_base64}
                }
            }
            response = requests.post(url, json=payload)
            result = response.json()

            preds = result["outputs"][0]["predictions"]["predictions"]
            preds_sorted = sorted(preds, key=lambda x: x['confidence'], reverse=True)

            with col2:
                st.subheader("Prediction Results")
                top = preds_sorted[0]
                conf = top['confidence'] * 100

                if conf >= 80:
                    st.success(top['class'].replace('_', ' ') + " - " + str(round(conf, 1)) + "% confidence")
                elif conf >= 50:
                    st.warning(top['class'].replace('_', ' ') + " - " + str(round(conf, 1)) + "% confidence")
                else:
                    st.error(top['class'].replace('_', ' ') + " - " + str(round(conf, 1)) + "% confidence (low certainty)")

                st.caption(CLASS_INFO.get(top['class'], ""))

                st.write("All class probabilities:")
                for p in preds_sorted:
                    class_name = p['class'].replace('_', ' ')
                    conf_val = p['confidence'] * 100
                    st.progress(p['confidence'], text=class_name + ": " + str(round(conf_val, 1)) + "%")

            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Image": image_name,
                "Prediction": top['class'].replace('_', ' '),
                "Confidence": str(round(conf, 1)) + "%"
            })

            report_lines = []
            report_lines.append("X-ray Quality Control Report")
            report_lines.append("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            report_lines.append("")
            report_lines.append("Image: " + str(image_name))
            report_lines.append("Prediction: " + top['class'].replace('_', ' '))
            report_lines.append("Confidence: " + str(round(conf, 1)) + "%")
            report_lines.append("")
            report_lines.append("All class probabilities:")
            for p in preds_sorted:
                class_name = p['class'].replace('_', ' ')
                conf_val = p['confidence'] * 100
                report_lines.append("- " + class_name + ": " + str(round(conf_val, 1)) + "%")

            report_text = "\n".join(report_lines)

            st.download_button(
                label="Download Report",
                data=report_text,
                file_name="xray_qc_report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error("Something went wrong: " + str(e))

if st.session_state.history:
    st.markdown("---")
    st.subheader("Session History")
    st.table(st.session_state.history)

st.markdown("---")
st.caption("Built by Zara Ashraf | Custom-trained Vision Transformer classification model.")
