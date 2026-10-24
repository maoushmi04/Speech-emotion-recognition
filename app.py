
import io
from pathlib import Path

import librosa
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models


# =========================================================
# 1. CONFIGURATION
# =========================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "emotion_recognition_model.pth"
)

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "pleasant_surprise",
    "sadness",
    "neutral",
]

SR = 16000
N_MELS = 128
MAX_LEN = 128

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


# =========================================================
# 2. MODEL ARCHITECTURE
# Must match the model used during training.
# =========================================================

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.vgg = models.vgg16(weights=None)

        self.vgg.classifier[6] = nn.Linear(
            self.vgg.classifier[6].in_features,
            num_classes,
        )

    def forward(self, x):
        return self.vgg(x)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Place emotion_recognition_model.pth "
            "in the same folder as app.py."
        )

    device = torch.device("cpu")

    model = EmotionRecognitionModel(len(EMOTIONS))

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device


# =========================================================
# 3. AUDIO PREPROCESSING
# =========================================================

def audio_bytes_to_tensor(audio_bytes):
    """Convert audio into the representation used in training."""

    y, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=SR,
        mono=True,
    )

    if len(y) == 0:
        raise ValueError("The audio recording is empty.")

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
    )

    db = librosa.power_to_db(mel, ref=np.max)

    db = np.clip(
        (db + 80.0) / 80.0,
        0.0,
        1.0,
    )

    # Pad or truncate to 128 time frames.
    if db.shape[1] < MAX_LEN:
        db = np.pad(
            db,
            (
                (0, 0),
                (0, MAX_LEN - db.shape[1]),
            ),
            mode="constant",
        )
    else:
        db = db[:, :MAX_LEN]

    # Convert to three channels for VGG16.
    x = torch.tensor(
        db,
        dtype=torch.float32,
    ).unsqueeze(0).repeat(3, 1, 1)

    # ImageNet normalization.
    x = (x - IMAGENET_MEAN) / IMAGENET_STD

    return x


# =========================================================
# 4. EMOTION PREDICTION
# =========================================================

def predict_audio(audio_bytes):
    model, device = load_model()

    x = audio_bytes_to_tensor(audio_bytes)

    # Add batch dimension.
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1)[0]

    index = int(probabilities.argmax().item())

    return (
        EMOTIONS[index],
        float(probabilities[index].item()),
        probabilities.cpu().numpy(),
    )


def display_prediction(emotion, confidence, probabilities):
    """Display prediction results in a reusable layout."""

    display_emotion = emotion.replace(
        "_", " "
    ).title()

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">PREDICTED EMOTION</div>
            <div class="result-emotion">{display_emotion}</div>
            <div class="result-note">
                Estimated from the supplied speech audio
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.metric(
        label="Model confidence score",
        value=f"{confidence:.2%}",
    )

    st.caption(
        "This score reflects the model's prediction, "
        "not a guarantee that the emotion is correct."
    )

    st.subheader("Emotion score breakdown")

    chart_data = {
        name.replace("_", " ").title(): float(score)
        for name, score in zip(EMOTIONS, probabilities)
    }

    st.bar_chart(chart_data)


# =========================================================
# 5. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 6. CUSTOM UI STYLING
# =========================================================

st.markdown(
    """
    <style>
    /* Overall background */
    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                #1c3157 0%,
                transparent 38%
            ),
            linear-gradient(
                135deg,
                #081225 0%,
                #101c35 55%,
                #152644 100%
            );
        color: #eef3ff;
    }

    /* Page spacing */
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Headings */
    h1 {
        color: #f5f7ff !important;
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #e5ecff !important;
    }

    p, label, li {
        color: #d5e0f7;
    }

    /* Hero section */
    .hero {
        padding: 26px;
        border: 1px solid #30476e;
        border-radius: 20px;
        background: rgba(21, 39, 69, 0.76);
        margin-bottom: 20px;
    }

    .hero-tag {
        display: inline-block;
        color: #a9c4ff;
        background: rgba(101, 124, 230, 0.15);
        border: 1px solid #425a91;
        border-radius: 30px;
        padding: 6px 12px;
        font-size: 0.8rem;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        color: #b9c9e8;
        font-size: 1rem;
        line-height: 1.7;
    }

    /* Feature cards */
    .feature-card {
        border: 1px solid #344c76;
        border-radius: 18px;
        padding: 20px;
        min-height: 125px;
        background: linear-gradient(
            145deg,
            rgba(36, 58, 95, 0.88),
            rgba(20, 35, 63, 0.9)
        );
        margin-bottom: 12px;
    }

    .feature-icon {
        font-size: 1.8rem;
        margin-bottom: 10px;
    }

    .feature-title {
        font-size: 1.2rem;
        font-weight: 750;
        color: #f3f6ff;
        margin-bottom: 6px;
    }

    .feature-description {
        color: #b7c7e6;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* Upload widget */
    div[data-testid="stFileUploader"] {
        background: rgba(35, 55, 88, 0.48);
        border: 1px dashed #8298d5;
        border-radius: 14px;
        padding: 16px;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(26, 43, 72, 0.7);
        border: 1px dashed #7189c4;
        border-radius: 12px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        min-height: 46px;
        border-radius: 11px;
        border: 1px solid #7184ef;
        background: linear-gradient(
            90deg,
            #655ce0,
            #4279ed
        );
        color: white;
        font-weight: 700;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #b4c3ff;
        filter: brightness(1.12);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #c5d4f3;
        font-weight: 700;
        padding: 14px 18px;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #9ebaff;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #8b7cff;
    }

    /* Results */
    .result-card {
        background: linear-gradient(
            135deg,
            rgba(28, 67, 81, 0.9),
            rgba(22, 48, 71, 0.95)
        );
        border: 1px solid #3e938d;
        border-radius: 18px;
        padding: 24px;
        margin-top: 18px;
        margin-bottom: 16px;
    }

    .result-label {
        color: #9fe8d7;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .result-emotion {
        color: #f3fffb;
        font-size: 2.1rem;
        font-weight: 800;
        margin-top: 8px;
    }

    .result-note {
        color: #b7d8d8;
        margin-top: 5px;
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(31, 51, 83, 0.85);
        border: 1px solid #354f7a;
        border-radius: 12px;
        padding: 16px;
    }

    div[data-testid="stMetricValue"] {
        color: #77e7c0;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #91a7cf;
        font-size: 0.85rem;
        padding: 30px 0 8px;
        border-top: 1px solid #263b5e;
        margin-top: 35px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 7. HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">
            DEEP LEARNING • VGG16 TRANSFER LEARNING
        </div>
        <h1>🎙️ Speech Emotion Recognition</h1>
        <div class="hero-subtitle">
            Explore the emotions expressed in speech.
            Upload an existing audio file or record your
            voice, then let the trained model estimate
            the emotion.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "🔒 Audio is processed in memory for prediction. "
    "This app does not deliberately save your uploaded "
    "or recorded audio to a file."
)


# =========================================================
# 8. UPLOAD AND RECORD SECTIONS
# =========================================================

tab_upload, tab_record = st.tabs(
    ["📁  Upload Audio", "🎤  Record Voice"]
)


# ---------------------------------------------------------
# TAB 1: UPLOAD AN AUDIO FILE
# ---------------------------------------------------------

with tab_upload:

    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📂</div>
            <div class="feature-title">Upload your audio</div>
            <div class="feature-description">
                Select a WAV audio file from your computer.
                Listen to it before running the prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "⬇️ Choose a WAV file from your computer",
        type=["wav"],
        help="Select an existing WAV recording.",
        key="uploaded_audio",
    )

    if uploaded_file is not None:

        audio_bytes = uploaded_file.getvalue()

        st.success(
            f"File selected: {uploaded_file.name}"
        )

        st.audio(
            audio_bytes,
            format="audio/wav",
        )

        if st.button(
            "✨ Predict Emotion from Uploaded Audio",
            key="predict_upload",
        ):
            try:
                with st.spinner(
                    "Analyzing audio with the trained model..."
                ):
                    emotion, confidence, probabilities = (
                        predict_audio(audio_bytes)
                    )

                display_prediction(
                    emotion,
                    confidence,
                    probabilities,
                )

            except Exception as exc:
                st.error(
                    f"Prediction failed: {exc}"
                )

    else:
        st.caption(
            "👆 The upload box above is where you select "
            "an existing WAV audio file."
        )


# ---------------------------------------------------------
# TAB 2: RECORD USING YOUR MICROPHONE
# ---------------------------------------------------------

with tab_record:

    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🎧</div>
            <div class="feature-title">Record your voice</div>
            <div class="feature-description">
                Use your device's microphone to record a
                short speech sample, then predict its
                estimated emotion.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "Allow microphone access in your browser if asked."
    )

    recorded_audio = st.audio_input(
        "🎙️ Click to record your voice",
        sample_rate=SR,
        key="microphone_audio",
    )

    if recorded_audio is not None:

        audio_bytes = recorded_audio.getvalue()

        st.success(
            "Recording received. Listen before predicting."
        )

        st.audio(
            audio_bytes,
            format="audio/wav",
        )

        if st.button(
            "✨ Predict Emotion from Recorded Voice",
            key="predict_record",
        ):
            try:
                with st.spinner(
                    "Analyzing your voice..."
                ):
                    emotion, confidence, probabilities = (
                        predict_audio(audio_bytes)
                    )

                display_prediction(
                    emotion,
                    confidence,
                    probabilities,
                )

            except Exception as exc:
                st.error(
                    f"Prediction failed: {exc}"
                )


# =========================================================
# 9. FOOTER
# =========================================================

st.markdown(
    """
    <div class="app-footer">
        Speech Emotion Recognition<br>
        TESS Dataset &nbsp; • &nbsp;
        VGG16 Transfer Learning &nbsp; • &nbsp;
        7 Emotion Classes
    </div>
    """,
    unsafe_allow_html=True,
)




#python -m streamlit run app.py