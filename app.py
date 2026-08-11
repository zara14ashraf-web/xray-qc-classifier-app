import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(page_title="X-ray Quality Control Classifier", page_icon="🔬", layout="centered")

API_KEY = st.secrets["ROBOFLOW_API_KEY"]
WORKSPACE = "zara-ashraf"
WORKFLOW_ID = "xray-qc-classifier-vxray-qc-classifier-evd8y-1-vit-base-patch16-224-in21k-t1-logic"

CLASS_INFO = {
    "Good_Quality": "The X-ray meets quality standards — no visible defects.",
    "Blur": "The image shows motion blur or focus issues.",
    "Exposure_Error": "The image is over-exposed or under-exposed.",
    "Foreign_Artifact": "An unexpected object (metal, jewelry, hardware) is visible in the image.",
}

# ---- SIDEBAR ----
with st.sidebar:
    st.header("About this project")
    st.write("This tool automatically checks radiograph images for common quality-control issues using a custom-trained Vision Transformer (ViT) model.")
    st.subheader("Detected Classes")
    for cls, desc in CLASS_INFO.items():
        st.markdown(f"**{cls.replace('_', ' ')}**")
        st.caption(desc)
    st.markdown("---")
    st.caption("Developed by Zara Ashraf")

# ---- HEADER ----
st.markdown("<h1 style='text-align: center;'>🔬 X-ray Quality Control Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-powered radiograph quality assessment tool</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 0.9em;'>by Zara Ashraf</p>", unsafe_allow_html=True)
st.markdown("---")
st.info("⚠️ This is a demonstration prototype trained on a limited, self-curated dataset. It's a proof-of-concept that shows the approach works — accuracy will continue to improve as the dataset grows and the model is retrained.")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "jfif"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    with st.spinner("Running quality checks on your image..."):
        try:
            url = f"https://serverless.roboflow.com/infer/workflows/{WORKSPACE}/{WORKFLOW_ID}"
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
                    st.success(f"**{top['class'].replace('_', ' ')}** — {conf:.1f}% confidence")
                elif conf >= 50:
                    st.warning(f"**{top['class'].replace('_', ' ')}** — {conf:.1f}% confidence")
                else:
                    st.error(f"**{top['class'].replace('_', ' ')}** — {conf:.1f}% confidence (low certainty)")

                st.caption(CLASS_INFO.get(top['class'], ""))

                st.write("All class probabilities:")
                for p in preds_sorted:
                    st.progress(p['confidence'], text=f"{p['class'].replace('_', ' ')}: {p['confidence']*100:.1f}%")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.markdown("---")
st.caption("Built by Zara Ashraf | Custom-trained Vision Transformer classification model.")
