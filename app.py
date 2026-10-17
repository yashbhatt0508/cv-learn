
from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AgeLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

HF_REPO_ID = "Nekowow/AgeLens_AI"
CHECKPOINT_FILENAME = "resnet18_age_finetuned.pth"

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(
    repo_id: str,
    filename: str,
    token: str,
) -> tuple[nn.Module, torch.device]:
    """Download the checkpoint and load the trained ResNet18."""

    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=token,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The checkpoint does not contain a state dictionary."
        )

    # Handle checkpoints saved using DataParallel.
    if checkpoint and all(
        key.startswith("module.") for key in checkpoint
    ):
        checkpoint = {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
        }

    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    return model, device


# --------------------------------------------------
# IMAGE PROCESSING AND PREDICTION
# --------------------------------------------------

def decode_image(image_bytes: bytes) -> Image.Image:
    """Decode the uploaded image and convert it to RGB."""

    with Image.open(BytesIO(image_bytes)) as source:
        source.load()
        return source.convert("RGB")


def predict_age(
    image: Image.Image,
    model: nn.Module,
    device: torch.device,
) -> float:
    """Predict the apparent age from a single image."""

    input_tensor = IMAGE_TRANSFORM(
        image.convert("RGB")
    ).unsqueeze(0).to(device)

    if tuple(input_tensor.shape) != (1, 3, 224, 224):
        raise ValueError(
            "The image could not be prepared for prediction."
        )

    with torch.inference_mode():
        output = model(input_tensor)

    if output.numel() != 1:
        raise ValueError(
            "The model returned an unexpected output shape."
        )

    age = float(output.reshape(-1)[0].item())

    if not torch.isfinite(torch.tensor(age)):
        raise ValueError(
            "The model returned an invalid age estimate."
        )

    # Keep the displayed estimate within a plausible human-age range.
    return max(0.0, min(120.0, age))


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #f5f4fb;
            --muted: #a8a8bb;
            --accent: #a78bfa;
            --accent2: #67e8f9;
            --panel: #151522;
            --border: rgba(255,255,255,0.10);
        }

        .stApp {
            background:
                radial-gradient(
                    ellipse at 15% 0%,
                    rgba(124,58,237,0.16),
                    transparent 38%
                ),
                radial-gradient(
                    ellipse at 90% 15%,
                    rgba(34,211,238,0.09),
                    transparent 32%
                ),
                #0b0b12;
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(11,11,18,0.85);
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: var(--ink) !important;
            letter-spacing: -0.035em;
        }

        p, li, label {
            color: #d1d1df;
        }

        .hero {
            padding: 2.2rem 0 1.7rem 0;
        }

        .eyebrow {
            display: inline-block;
            color: #c4b5fd;
            border: 1px solid rgba(167,139,250,0.30);
            background: rgba(139,92,246,0.10);
            padding: 0.38rem 0.8rem;
            border-radius: 999px;
            font-size: 0.78rem;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: clamp(2.6rem, 6vw, 4.5rem);
            line-height: 1.05;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.065em;
            margin: 0;
        }

        .gradient-text {
            background: linear-gradient(
                90deg, #c4b5fd, #67e8f9
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            color: #aaaabe;
            font-size: 1.08rem;
            max-width: 680px;
            line-height: 1.8;
            margin-top: 1.1rem;
        }

        .panel {
            background: rgba(21,21,34,0.82);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.35rem;
            min-height: 100%;
        }

        .panel-label {
            color: #a8a8bb;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            font-size: 0.74rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .result-card {
            background: linear-gradient(
                145deg,
                rgba(139,92,246,0.15),
                rgba(34,211,238,0.06)
            );
            border: 1px solid rgba(167,139,250,0.28);
            border-radius: 20px;
            padding: 1.7rem;
            margin-top: 1rem;
        }

        .result-number {
            font-size: clamp(3.3rem, 8vw, 5rem);
            line-height: 1.1;
            letter-spacing: -0.06em;
            font-weight: 800;
            color: #ffffff;
        }

        .result-caption {
            color: #bcbcd0;
            margin-top: 0.55rem;
        }

        .muted {
            color: #a8a8bb;
            font-size: 0.92rem;
            line-height: 1.7;
        }

        .step-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2rem;
            height: 2rem;
            border-radius: 10px;
            color: #ddd6fe;
            background: rgba(139,92,246,0.16);
            border: 1px solid rgba(167,139,250,0.25);
            font-weight: 700;
            margin-right: 0.55rem;
        }

        div.stButton > button {
            border: 0;
            border-radius: 12px;
            min-height: 3rem;
            font-weight: 700;
            color: #10101a;
            background: linear-gradient(
                100deg, #c4b5fd, #67e8f9
            );
            transition: transform 0.15s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-1px);
            color: #10101a;
            border: 0;
        }

        [data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.025);
            border: 1px dashed rgba(167,139,250,0.45);
            border-radius: 15px;
            padding: 0.5rem;
        }

        [data-testid="stFileUploader"] section {
            border: 0;
        }

        [data-testid="stImage"] img {
            border-radius: 14px;
        }

        .footer {
            border-top: 1px solid var(--border);
            margin-top: 3rem;
            padding-top: 1.4rem;
            text-align: center;
            color: #88889d;
            font-size: 0.82rem;
            line-height: 1.8;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main() -> None:
    inject_styles()

    # Header
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">✦ Computer Vision · Deep Learning</div>
            <div class="hero-title">
                See beyond the <span class="gradient-text">pixels.</span>
            </div>
            <div class="hero-subtitle">
                Meet AgeLens AI — an image-regression model that
                estimates apparent age from a facial photograph
                using a fine-tuned ResNet18 convolutional neural network.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Model configuration
    try:
        hf_token = st.secrets.get("HF_TOKEN", "")
    except Exception:
        hf_token = ""

    if not hf_token:
        st.error(
            "Hugging Face access token not found. Add HF_TOKEN "
            "to Streamlit Secrets before using the app."
        )
        st.code('HF_TOKEN = "your_read_token_here"', language="toml")
        st.caption(
            "For local testing, place this in "
            ".streamlit/secrets.toml. Never commit that file or "
            "your token to GitHub."
        )
        st.stop()

    # Upload and prediction layout
    upload_col, result_col = st.columns(
        [1, 1],
        gap="large",
    )

    with upload_col:
        st.markdown(
            '<div class="panel-label">01 / Upload an image</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Your image")
        st.markdown(
            '<p class="muted">Upload a clear, front-facing facial '
            'photograph for the best chance of a useful estimate.</p>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPG, JPEG, PNG and WEBP.",
        )

        image = None

        if uploaded_file is not None:
            try:
                image = decode_image(uploaded_file.getvalue())
                st.image(
                    image,
                    caption="Uploaded image",
                    use_container_width=True,
                )
                st.caption(
                    f"Image dimensions: {image.width} × {image.height}"
                )
            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
            ):
                st.error(
                    "This image could not be opened. "
                    "Try another JPG, PNG or WEBP image."
                )
                image = None

        predict_clicked = st.button(
            "✦  Estimate apparent age",
            use_container_width=True,
            disabled=image is None,
        )

    with result_col:
        st.markdown(
            '<div class="panel-label">02 / AI estimation</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Prediction")

        if predict_clicked and image is not None:
            try:
                with st.spinner(
                    "Loading the model and analysing the image..."
                ):
                    model, device = load_model(
                        HF_REPO_ID,
                        CHECKPOINT_FILENAME,
                        hf_token,
                    )
                    predicted_age = predict_age(
                        image,
                        model,
                        device,
                    )

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="panel-label">
                            Estimated apparent age
                        </div>
                        <div class="result-number">
                            {predicted_age:.1f}
                            <span style="font-size:1.4rem;
                                         letter-spacing:0;">
                                years
                            </span>
                        </div>
                        <div class="result-caption">
                            AI-generated estimate from the uploaded image
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.success("Prediction completed successfully.")

                st.markdown("#### About this result")
                st.markdown(
                    """
                    This number is a model prediction, not a verified
                    biological or chronological age. Individual results
                    can be inaccurate, and performance can vary across
                    age groups, lighting, image quality and demographics.
                    """
                )

                st.caption(
                    f"Model: ResNet18 · Device: {device.type.upper()}"
                )

            except Exception as exc:
                LOGGER.exception("Age prediction failed.")

                st.error(
                    "The prediction could not be completed."
                )

                # Keep credentials and internal details out of the UI.
                if "Repository Not Found" in str(exc) or (
                    "401" in str(exc)
                    or "403" in str(exc)
                    or "gated repo" in str(exc).lower()
                ):
                    st.warning(
                        "Hugging Face access failed. Check that the "
                        "repository ID is correct, the model file exists, "
                        "and your read token has access to the private repo."
                    )
                elif isinstance(exc, FileNotFoundError):
                    st.warning(
                        "The model checkpoint could not be found in "
                        "the Hugging Face repository."
                    )
                elif isinstance(exc, RuntimeError):
                    st.warning(
                        "The model could not load the checkpoint. "
                        "Confirm that it matches the fine-tuned ResNet18 "
                        "architecture."
                    )
                else:
                    st.warning(
                        "Check the Streamlit app logs for the technical "
                        "error details. Do not share your access token."
                    )

        else:
            st.markdown(
                """
                <div class="result-card">
                    <div class="panel-label">Your result will appear here</div>
                    <div style="font-size:2rem; margin:1rem 0;">
                        ◌
                    </div>
                    <div style="color:#bcbcd0; line-height:1.8;">
                        Upload a photograph, then select
                        <b>Estimate apparent age</b> to run the model.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### What happens behind the scenes?")
            st.markdown(
                """
                - The image is converted to RGB and resized to 224 × 224.
                - ImageNet normalization prepares the input for ResNet18.
                - The fine-tuned model produces a numerical age estimate.
                """
            )

    # How it works
    st.divider()
    st.markdown("## Built with computer vision")
    st.markdown(
        '<p class="muted">A compact look at the inference pipeline.</p>',
        unsafe_allow_html=True,
    )

    step1, step2, step3 = st.columns(3, gap="medium")

    with step1:
        st.markdown(
            """
            <div class="panel">
                <div class="step-number">1</div>
                <h3>Preprocess</h3>
                <p class="muted">
                    Convert the image to RGB, resize it to 224 × 224,
                    and normalize its pixel values.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step2:
        st.markdown(
            """
            <div class="panel">
                <div class="step-number">2</div>
                <h3>Extract features</h3>
                <p class="muted">
                    ResNet18 convolutional layers process visual
                    patterns learned from images.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step3:
        st.markdown(
            """
            <div class="panel">
                <div class="step-number">3</div>
                <h3>Regress an age</h3>
                <p class="muted">
                    A final linear layer outputs a continuous numerical
                    estimate rather than an age category.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Model information
    with st.expander("Model details & responsible use"):
        st.markdown(
            """
            **Architecture:** ResNet18 with a custom regression head.

            **Framework:** PyTorch and Torchvision.

            **Input:** One facial image resized to 224 × 224 RGB.

            **Output:** One continuous age estimate in years.

            **Training dataset:** UTKFace.

            **Important limitations:** This is an experimental machine
            learning project. Predictions can be wrong and may perform
            differently across populations and image conditions. Do not
            use its outputs for identity verification, eligibility
            decisions, medical decisions or other high-stakes purposes.
            Use photographs only with appropriate permission.
            """
        )

    # Footer
    st.markdown(
        """
        <div class="footer">
            <b>AgeLens AI</b> · A computer vision learning project<br>
            Powered by PyTorch, Torchvision and Streamlit<br>
            Predictions are estimates and may be inaccurate.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
