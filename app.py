import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
from datetime import datetime
import os
import requests


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="X-ray Quality Control Classifier",
    page_icon="🩻",
    layout="centered"
)


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

MODEL_PATH = "best_xray_qc_vit.pth"

MODEL_URL = (
    "https://huggingface.co/zara14ashraf/xray-qc-vit/"
    "resolve/main/best_xray_qc_vit.pth"
)

if not os.path.exists(MODEL_PATH):

    with st.spinner("Loading AI model..."):

        response = requests.get(
            MODEL_URL,
            timeout=300
        )

        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)


CLASS_NAMES = [
    "Blur",
    "Exposure_Error",
    "Foreign_Artifact",
    "Good_Quality"
]

CLASS_INFO = {
    "Good_Quality": "The X-ray meets quality standards - no visible defects.",
    "Blur": "The image shows motion blur or focus issues.",
    "Exposure_Error": "The image is over-exposed or under-exposed.",
    "Foreign_Artifact": "An unexpected object (metal, jewelry, hardware) is visible in the image.",
}


# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

device = torch.device("cpu")


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=4
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, checkpoint


model, checkpoint = load_model()


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------

def predict_image(image):

    image = image.convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    predicted_index = probabilities.argmax().item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    return predicted_class, probabilities


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
        "quality-control issues using a custom-trained Vision Transformer "
        "(ViT) model."
    )

    st.subheader("Detected Classes")

    for number, (cls, desc) in enumerate(
        CLASS_INFO.items(),
        start=1
    ):

        class_name = cls.replace(
            "_",
            " "
        )

        st.markdown(
            f"**{number:02d} — {class_name}**"
        )

        st.caption(desc)

    st.markdown("---")

    st.caption(
        "Developed by Zara Ashraf"
    )


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------

st.markdown(
    "<h1 style='text-align: center;'>"
    "X-ray Quality Control Classifier"
    "</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "AI-assisted assessment of common radiographic "
    "image-quality issues"
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
# INFORMATION BOX
# ---------------------------------------------------------

st.info(
    "This is a demonstration prototype trained on a limited, self-curated dataset. "
    "It's a proof-of-concept that shows the approach works, accuracy will continue "
    "to improve as the dataset grows and the model is retrained."
)


# ---------------------------------------------------------
# WHAT DOES THIS TOOL ASSESS?
# ---------------------------------------------------------

st.subheader(
    "What does this tool assess?"
)

assessment_cols = st.columns(4)

assessment_items = [
    ("✓", "Good Quality"),
    ("✓", "Motion Blur"),
    ("✓", "Exposure Error"),
    ("✓", "Foreign Artifact"),
]

for col, (symbol, text) in zip(
    assessment_cols,
    assessment_items
):

    with col:

        st.markdown(
            f"**{symbol} {text}**"
        )


# ---------------------------------------------------------
# WHY I DEVELOPED THIS TOOL
# ---------------------------------------------------------

with st.expander(
    "Why I Developed This Tool"
):

    st.write(
        "In medical imaging, obtaining a diagnostic image is not only "
        "about producing an X-ray. Image quality also matters because "
        "positioning, motion, exposure, and external artifacts can affect "
        "how an examination is interpreted."
    )

    st.write(
        "I developed this project to explore how artificial intelligence "
        "could be used as a supportive quality-control tool for "
        "radiographic images. The goal is not to replace radiographers "
        "or radiologists, but to investigate whether a trained AI model "
        "can identify common image-quality issues and provide an "
        "additional layer of support before image interpretation."
    )

    st.markdown(
        "**The idea behind the project is simple: instead of using AI "
        "only to detect disease, can we also use it to help ensure that "
        "the image itself is suitable for interpretation?**"
    )

    st.write(
        "This project allowed me to combine my clinical background in "
        "Medical Imaging Technology with my interest in artificial "
        "intelligence and medical imaging research."
    )


# ---------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------

with st.expander(
    "How it works"
):

    st.markdown("### 1 — Upload")

    st.write(
        "A radiographic image is uploaded to the application."
    )

    st.markdown("### 2 — Preprocessing")

    st.write(
        "The image is resized and normalized using the same "
        "preprocessing approach used for the trained model."
    )

    st.markdown("### 3 — AI Classification")

    st.write(
        "The locally stored Vision Transformer (ViT) model "
        "classifies the image into one of four quality categories."
    )

    st.markdown("### 4 — Result")

    st.write(
        "The application displays the predicted category and "
        "confidence scores."
    )


# ---------------------------------------------------------
# ABOUT THE MODEL
# ---------------------------------------------------------

with st.expander(
    "About the Model"
):

    model_col1, model_col2 = st.columns(2)

    with model_col1:

        st.markdown(
            "**Model architecture**"
        )

        st.write(
            "Vision Transformer (ViT)"
        )

        st.markdown(
            "**Task**"
        )

        st.write(
            "Multi-class image classification"
        )

    with model_col2:

        st.markdown(
            "**Classes**"
        )

        st.write("4")

        st.markdown(
            "**Inference**"
        )

        st.write(
            "Local model inference"
        )

    st.markdown(
        "**Best validation accuracy**"
    )

    st.write(
        f"{checkpoint['best_val_accuracy'] * 100:.2f}%"
    )

    st.markdown(
        "**Application**"
    )

    st.write(
        "Radiographic image-quality assessment"
    )


# ---------------------------------------------------------
# SAMPLE IMAGES
# ---------------------------------------------------------

st.subheader(
    "Try a sample image"
)


def find_sample(cloud_path, local_path):

    if os.path.exists(cloud_path):
        return cloud_path

    if os.path.exists(local_path):
        return local_path

    return None


sample_files = [
    (
        "Blur",
        find_sample(
            "streamlit/sample 1.jpg",
            os.path.join(
                "Xray_QC_ViT",
                "test",
                "Blur",
                os.listdir(
                    os.path.join(
                        "Xray_QC_ViT",
                        "test",
                        "Blur"
                    )
                )[0]
            ) if os.path.exists(
                os.path.join(
                    "Xray_QC_ViT",
                    "test",
                    "Blur"
                )
            ) else None
        )
    ),

    (
        "Exposure Error",
        find_sample(
            "streamlit/sample 2.png",
            os.path.join(
                "Xray_QC_ViT",
                "test",
                "Exposure_Error",
                os.listdir(
                    os.path.join(
                        "Xray_QC_ViT",
                        "test",
                        "Exposure_Error"
                    )
                )[0]
            ) if os.path.exists(
                os.path.join(
                    "Xray_QC_ViT",
                    "test",
                    "Exposure_Error"
                )
            ) else None
        )
    ),

    (
        "Foreign Artifact",
        find_sample(
            "streamlit/sample 3.png",
            os.path.join(
                "Xray_QC_ViT",
                "test",
                "Foreign_Artifact",
                "Cofield-59_png.rf.3c3c636c6b1e2be94f07de66cb644f08.jpg"
            )
        )
    ),

    (
        "Good Quality",
        find_sample(
            "streamlit/sample 4.jpeg",
            os.path.join(
                "Xray_QC_ViT",
                "test",
                "Good_Quality",
                "IM-0011-0001_jpeg.rf.6c6709c8368218bd35a19428ae8147ae.jpg"
            )
        )
    )
]


sample_cols = st.columns(4)

for i, col in enumerate(sample_cols):

    with col:

        if st.button(
            f"Sample {i + 1}",
            width="stretch"
        ):

            selected_path = sample_files[i][1]

            if selected_path is not None:

                st.session_state.selected_image_path = (
                    selected_path
                )

                st.rerun()

            else:

                st.error(
                    "Sample image not found."
                )


st.markdown("---")


# ---------------------------------------------------------
# UPLOAD SECTION
# ---------------------------------------------------------

st.subheader(
    "Upload an X-ray"
)

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

        image_source = Image.open(
            uploaded_file
        ).convert("RGB")

        image_name = uploaded_file.name

        st.session_state.selected_image_path = None

    except Exception:

        st.error(
            "The uploaded file could not be opened as an image."
        )


elif st.session_state.selected_image_path is not None:

    try:

        image_source = Image.open(
            st.session_state.selected_image_path
        ).convert("RGB")

        sample_path = (
            st.session_state.selected_image_path
        )

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

        st.warning(
            "Sample image not found."
        )


# ---------------------------------------------------------
# IMAGE ANALYSIS
# ---------------------------------------------------------

if image_source is not None:

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image_source,
            caption="Selected X-ray",
            width="stretch"
        )

    # -----------------------------------------------------
    # LOCAL MODEL INFERENCE
    # -----------------------------------------------------

    with st.spinner(
        "Running quality checks on your image..."
    ):

        try:

            predicted_class, probabilities = (
                predict_image(image_source)
            )

            # ---------------------------------------------
            # AI ASSESSMENT
            # ---------------------------------------------

            with col2:

                st.subheader(
                    "AI Assessment"
                )

                predicted_index = (
                    probabilities.argmax().item()
                )

                conf = (
                    probabilities[predicted_index].item()
                    * 100
                )

                class_display_name = (
                    predicted_class.replace(
                        "_",
                        " "
                    )
                )

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

                st.markdown(
                    "**Interpretation**"
                )

                st.write(
                    CLASS_INFO.get(
                        predicted_class,
                        ""
                    )
                )

                st.caption(
                    "Confidence reflects the model's prediction "
                    "score and should not be interpreted as "
                    "clinical certainty."
                )


            # ---------------------------------------------
            # CLASS PROBABILITIES
            # ---------------------------------------------

            st.markdown("---")

            st.subheader(
                "Class Probabilities"
            )

            probability_data = []

            for class_name, probability in zip(
                CLASS_NAMES,
                probabilities
            ):

                confidence_value = (
                    probability.item()
                )

                confidence_percent = (
                    confidence_value * 100
                )

                probability_data.append(
                    (
                        class_name,
                        confidence_value,
                        confidence_percent
                    )
                )

            probability_data.sort(
                key=lambda x: x[1],
                reverse=True
            )

            for (
                class_name,
                confidence_value,
                confidence_percent
            ) in probability_data:

                display_name = (
                    class_name.replace(
                        "_",
                        " "
                    )
                )

                prob_col1, prob_col2 = (
                    st.columns([4, 1])
                )

                with prob_col1:

                    st.markdown(
                        f"**{display_name}**"
                    )

                with prob_col2:

                    st.markdown(
                        f"**{confidence_percent:.1f}%**"
                    )

                st.progress(
                    confidence_value
                )


            # ---------------------------------------------
            # SESSION HISTORY
            # ---------------------------------------------

            st.session_state.history.append(
                {
                    "Time": datetime.now().strftime(
                        "%H:%M:%S"
                    ),
                    "Image": image_name,
                    "Prediction": class_display_name,
                    "Confidence": f"{conf:.1f}%"
                }
            )


            # ---------------------------------------------
            # DOWNLOAD REPORT
            # ---------------------------------------------

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

            for (
                class_name,
                confidence_value,
                confidence_percent
            ) in probability_data:

                display_name = (
                    class_name.replace(
                        "_",
                        " "
                    )
                )

                report_lines.append(
                    f"{display_name}: "
                    f"{confidence_percent:.1f}%"
                )

            report_lines.append("")

            report_lines.append(
                "Model: Custom Vision Transformer (ViT)"
            )

            report_lines.append(
                "Inference: Local model"
            )

            report_lines.append(
                "Application: Radiographic "
                "image-quality assessment"
            )

            report_lines.append("")

            report_lines.append(
                "Note: This AI-generated assessment is "
                "intended for research and prototype "
                "demonstration purposes."
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


        except Exception as e:

            st.error(
                "Something went wrong during model inference: "
                + str(e)
            )


# ---------------------------------------------------------
# SESSION HISTORY
# ---------------------------------------------------------

if st.session_state.history:

    st.markdown("---")

    st.subheader(
        "Session History"
    )

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
    "For research and educational purposes only. "
    "This tool is not intended for clinical diagnosis "
    "or to replace professional medical judgment."
)
