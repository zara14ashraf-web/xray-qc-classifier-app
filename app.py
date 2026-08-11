import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(page_title="X-ray Quality Control Classifier", page_icon="🔬", layout="centered")

st.markdown("<h1 style='text-align: center;'>🔬 X-ray Quality Control Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-powered radiograph quality assessment tool</p>", unsafe_allow_html=True)
st.markdown("---")

API_KEY = st.secrets["ROBOFLOW_API_KEY"]
WORKSPACE = "zara-ashraf"
WORKFLOW_ID = "xray-qc-classifier-vxray-qc-classifier-evd8y-1-vit-base-patch16-224-in21k-t1-logic"

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    with st.spinner("Analyzing image..."):
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
                st.success(f"**{top['class']}** — {top['confidence']*100:.1f}% confidence")
                st.write("All class probabilities:")
                for p in preds_sorted:
                    st.progress(p['confidence'], text=f"{p['class']}: {p['confidence']*100:.1f}%")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.write(result if 'result' in dir() else "No response received")

st.markdown("---")
st.caption("Built with a custom-trained Vision Transformer classification model.")
