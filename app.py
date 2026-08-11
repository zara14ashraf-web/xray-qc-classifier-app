import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="X-ray Quality Control Classifier", page_icon="🩻", layout="centered")

st.markdown("<h1 style='text-align: center;'>🩻 X-ray Quality Control Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-powered radiograph quality assessment tool</p>", unsafe_allow_html=True)
st.markdown("---")

API_KEY = st.secrets["ROBOFLOW_API_KEY"]
WORKSPACE = "zara-ashraf"
WORKFLOW_ID = "xray-qc-classifier-vxray-qc-classifier-evd8y-1-vit-base-patch16-224-in21k-t1-logic"

@st.cache_resource
def get_client():
    return InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=API_KEY)

client = get_client()

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.convert("RGB").save(tmp.name)
        tmp_path = tmp.name

    with st.spinner("Analyzing image..."):
        try:
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,
                images={"image": tmp_path},
                use_cache=True
            )
            preds = result[0]['predictions']['predictions']
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
        finally:
            os.remove(tmp_path)

st.markdown("---")
st.caption("Built with a custom-trained Vision Transformer classification model.")
