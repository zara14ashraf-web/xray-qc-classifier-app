import streamlit as st
import requests
import base64
from PIL import Image
import io
from datetime import datetime


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="X-ray Quality Control Classifier",
    page_icon="🩻",
    layout="centered"
)


# ---------------------------------------------------------
# ROBOFLOW CONFIGURATION
# ---------------------------------------------------------

API_KEY = st.secrets["ROBOFLOW_API_KEY"]

WORKSPACE = "zara-ashraf"

WORKFLOW_ID = "xray-qc-classifier-vxray-qc-classifier-qq9jj-1-vit-base-patch16-224-in21k-t1-logic"


# ---------------------------------------------------------
# CLASS INFORMATION
# ---------------------------------------------------------

CLASS_INFO = {
    "Good_Quality": "The X-ray meets quality standards - no visible defects.",
    "Blur": "The image shows motion blur or focus issues.",
    "Exposure_Error": "The image is over-exposed or under-exposed.",
    "Foreign_Artifact": "An unexpected object (metal, jewelry, hardware) is visible in the image.",
}


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "selected_image_path" not in st.session_state:
    st.session_state.selected_image_path = None


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("About this project")

    st.write(
        "This tool automatically checks radiograph images for common "
        "quality-control issues using a custom-trained Vision Transformer (ViT) model."
    )

    st.subheader("Detected Classes")

    for number, (cls, desc) in enumerate(CLASS_INFO.items(), start=1):

        class_name = cls.replace("_", " ")

        st.markdown(
            f"**{number:02d} — {class_name}**"
        )

        st.caption(desc)

    st.markdown("---")

    st.caption("Developed by Zara Ashraf")


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------

st.markdown(
    "<h1 style='text-align: center;'>X-ray Quality Control Classifier</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "AI-assisted assessment of common radiographic image-quality issues"
    "</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.9em;'>"
    "by Zara Ashraf"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")


# ---------------------------------------------------------
# ORIGINAL BLUE INFORMATION BOX
# ---------------------------------------------------------

st.info(
    "This is a demonstration prototype trained on a limited, self-curated dataset. "
    "It's a proof-of-concept that shows the approach works, accuracy will continue "
    "to improve as the dataset grows and the model is retrained."
)


# ---------------------------------------------------------
# WHAT DOES THIS TOOL ASSESS?
# ---------------------------------------------------------

st.subheader("What does this tool assess?")

assessment_cols = st.columns(4)

assessment_items = [
    ("✓", "Good Quality"),
    ("✓", "Motion Blur"),
    ("✓", "Exposure Error"),
    ("✓", "Foreign Artifact"),
]

for col, (symbol, text) in zip(assessment_cols, assessment_items):

    with col:
        st.markdown(f"**{symbol} {text}**")


# ---------------------------------------------------------
# WHY I DEVELOPED THIS TOOL
# ---------------------------------------------------------

with st.expander("Why I Developed This Tool"):

    st.write(
        "In medical imaging, obtaining a diagnostic image is not only about "
        "producing an X-ray. Image quality also matters because positioning, "
        "motion, exposure, and external artifacts can affect how an examination "
        "is interpreted."
    )

    st.write(
        "I developed this project to explore how artificial intelligence could "
        "be used as a supportive quality-control tool for radiographic images. "
        "The goal is not to replace radiographers or radiologists, but to "
        "investigate whether a trained AI model can identify common image-quality "
        "issues and provide an additional layer of support before image interpretation."
    )

    st.markdown(
        "**The idea behind the project is simple: instead of using AI only to "
        "detect disease, can we also use it to help ensure that the image itself "
        "is suitable for interpretation?**"
    )

    st.write(
        "This project allowed me to combine my clinical background in Medical "
        "Imaging Technology with my interest in artificial intelligence and "
        "medical imaging research."
    )


# ---------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------

with st.expander("How it works"):

    st.markdown("### 1 — Upload")

    st.write(
        "A radiographic image is uploaded to the application."
    )

    st.markdown("### 2 — Preprocessing")

    st.write(
        "The image is prepared and sent to the inference workflow."
    )

    st.markdown("### 3 — AI Classification")

    st.write(
        "A custom-trained Vision Transformer (ViT) classifies the image "
        "into one of four quality categories."
    )

    st.markdown("### 4 — Result")

    st.write(
        "The application displays the predicted category and confidence scores."
    )


# ---------------------------------------------------------
# ABOUT THE MODEL
# ---------------------------------------------------------

with st.expander("About the Model"):

    model_col1, model_col2 = st.columns(2)

    with model_col1:

        st.markdown("**Model architecture**")
        st.write("Vision Transformer (ViT)")

        st.markdown("**Task**")
        st.write("Multi-class image classification")

    with model_col2:

        st.markdown("**Classes**")
        st.write("4")

        st.markdown("**Inference**")
        st.write("Roboflow-hosted workflow")

    st.markdown("**Application**")
    st.write("Radiographic image-quality assessment")


# ---------------------------------------------------------
# SAMPLE IMAGES
# ---------------------------------------------------------

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

        if st.button(
            "Sample " + str(i + 1),
            use_container_width=True
        ):

            st.session_state.selected_image_path = sample_files[i]


st.markdown("---")


# ---------------------------------------------------------
# UPLOAD SECTION
# ---------------------------------------------------------

st.subheader("Upload an X-ray")

st.write(
    "Upload a radiograph to assess common image-quality issues."
)

uploaded_file = st.file_uploader(
    "Upload X-ray",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tiff",
        "webp",
        "jfif"
    ],
    help="Maximum file size: 200 MB"
)

st.caption(
    "Supported formats: JPG, JPEG, PNG, BMP, TIFF, WEBP, JFIF"
)


# ---------------------------------------------------------
# IMAGE SELECTION
# ---------------------------------------------------------

image_source = None
image_name = None

if uploaded_file is not None:

    try:

        image_source = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name

        st.session_state.selected_image_path = None

    except Exception:

        st.error("The uploaded file could not be opened as an image.")

elif st.session_state.selected_image_path is not None:

    try:

        image_source = Image.open(
            st.session_state.selected_image_path
        ).convert("RGB")

        sample_path = st.session_state.selected_image_path

        if "sample 1" in sample_path.lower():
            image_name = "Sample 1"

        elif "sample 2" in sample_path.lower():
            image_name = "Sample 2"

        elif "sample 3" in sample_path.lower():
            image_name = "Sample 3"

        elif "sample 4" in sample_path.lower():
            image_name = "Sample 4"

        else:
            image_name = sample_path

    except FileNotFoundError:

        st.warning("Sample image not found.")


# ---------------------------------------------------------
# IMAGE ANALYSIS
# ---------------------------------------------------------

if image_source is not None:

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image_source,
            caption="Selected X-ray",
            use_container_width=True
        )

    buffered = io.BytesIO()

    image_source.save(
        buffered,
        format="JPEG"
    )

    img_base64 = base64.b64encode(
        buffered.getvalue()
    ).decode("utf-8")


    # -----------------------------------------------------
    # ROBOFLOW INFERENCE
    # -----------------------------------------------------

    with st.spinner(
        "Running quality checks on your image..."
    ):

        try:

            url = (
                "https://serverless.roboflow.com/infer/workflows/"
                + WORKSPACE
                + "/"
                + WORKFLOW_ID
            )

            payload = {
                "api_key": API_KEY,
                "inputs": {
                    "image": {
                        "type": "base64",
                        "value": img_base64
                    }
                }
            }

            response = requests.post(
                url,
                json=payload,
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

            preds = (
                result["outputs"][0]
                ["predictions"]
                ["predictions"]
            )

            preds_sorted = sorted(
                preds,
                key=lambda x: x["confidence"],
                reverse=True
            )


            # -------------------------------------------------
            # AI ASSESSMENT
            # -------------------------------------------------

            with col2:

                st.subheader("AI Assessment")

                top = preds_sorted[0]

                predicted_class = top["class"]

                class_display_name = predicted_class.replace(
                    "_",
                    " "
                )

                conf = top["confidence"] * 100

                if conf >= 80:

                    st.success(
                        f"**{class_display_name.upper()}**\n\n"
                        f"**{conf:.1f}% confidence**"
                    )

                elif conf >= 50:

                    st.warning(
                        f"**{class_display_name.upper()}**\n\n"
                        f"**{conf:.1f}% confidence**"
                    )

                else:

                    st.error(
                        f"**{class_display_name.upper()}**\n\n"
                        f"**{conf:.1f}% confidence — low certainty**"
                    )

                st.markdown("**Interpretation**")

                st.write(
                    CLASS_INFO.get(
                        predicted_class,
                        ""
                    )
                )

                st.caption(
                    "Confidence reflects the model's prediction score "
                    "and should not be interpreted as clinical certainty."
                )


            # -------------------------------------------------
            # CLASS PROBABILITIES
            # -------------------------------------------------

            st.markdown("---")

            st.subheader("Class Probabilities")

            for p in preds_sorted:

                class_name = p["class"].replace(
                    "_",
                    " "
                )

                confidence_value = p["confidence"]

                confidence_percent = confidence_value * 100

                prob_col1, prob_col2 = st.columns(
                    [4, 1]
                )

                with prob_col1:

                    st.markdown(
                        f"**{class_name}**"
                    )

                with prob_col2:

                    st.markdown(
                        f"**{confidence_percent:.1f}%**"
                    )

                st.progress(
                    confidence_value
                )


            # -------------------------------------------------
            # SESSION HISTORY
            # -------------------------------------------------

            st.session_state.history.append(
                {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Image": image_name,
                    "Prediction": class_display_name,
                    "Confidence": f"{conf:.1f}%"
                }
            )


            # -------------------------------------------------
            # DOWNLOAD REPORT
            # -------------------------------------------------

            report_lines = []

            report_lines.append(
                "X-RAY IMAGE QUALITY ASSESSMENT"
            )

            report_lines.append(
                "=" * 40
            )

            report_lines.append(
                "Generated: "
                + datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            report_lines.append("")

            report_lines.append(
                "Image: "
                + str(image_name)
            )

            report_lines.append(
                "Assessment: "
                + class_display_name
            )

            report_lines.append(
                "Confidence: "
                + f"{conf:.1f}%"
            )

            report_lines.append("")

            report_lines.append(
                "CLASS PROBABILITIES"
            )

            report_lines.append(
                "-" * 25
            )

            for p in preds_sorted:

                class_name = p["class"].replace(
                    "_",
                    " "
                )

                confidence_percent = p["confidence"] * 100

                report_lines.append(
                    f"{class_name}: "
                    f"{confidence_percent:.1f}%"
                )

            report_lines.append("")

            report_lines.append(
                "Model: Custom Vision Transformer (ViT)"
            )

            report_lines.append(
                "Application: Radiographic image-quality assessment"
            )

            report_lines.append("")

            report_lines.append(
                "Note: This AI-generated assessment is intended "
                "for research and prototype demonstration purposes."
            )

            report_text = "\n".join(
                report_lines
            )

            st.download_button(
                label="Download Report",
                data=report_text,
                file_name=(
                    "xray_qc_report_"
                    + datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    + ".txt"
                ),
                mime="text/plain"
            )


        except requests.exceptions.RequestException as e:

            st.error(
                "Unable to connect to the inference service. "
                "Please try again."
            )

        except Exception as e:

            st.error(
                "Something went wrong: "
                + str(e)
            )


# ---------------------------------------------------------
# SESSION HISTORY
# ---------------------------------------------------------

if st.session_state.history:

    st.markdown("---")

    st.subheader("Session History")

    st.table(
        st.session_state.history
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Built by Zara Ashraf | "
    "Custom-trained Vision Transformer classification model."
)
st.caption(
    "For research and educational purposes only. This tool is not intended "
    "for clinical diagnosis or to replace professional medical judgment."
)
