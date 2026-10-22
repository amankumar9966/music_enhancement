
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="MusicEnhance AI",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #17152c);
    color: #ffffff;
}
h1, h2, h3 {
    color: #ffffff;
}
p, label {
    color: #d1d5e8;
}
section[data-testid="stFileUploader"] {
    background: #191d32;
    border: 1px dashed #8b5cf6;
    border-radius: 14px;
    padding: 12px;
}
div.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #4f46e5);
    color: white;
    border: 0;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 600;
    width: 100%;
}
div.stButton > button:hover {
    border: 1px solid #c4b5fd;
    color: white;
}
div[data-testid="stMetric"] {
    background: #191d32;
    padding: 15px;
    border-radius: 12px;
}


h1, h2, h3 {
    letter-spacing: -0.5px;
}

[data-testid="stSidebar"] {
    background: #101426;
}

div[data-testid="stExpander"] {
    background: #191d32;
    border: 1px solid #343653;
    border-radius: 12px;
}

div[data-testid="stFileUploader"] section {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HELPER FUNCTIONS ----------------

def get_audio_info(file_path):
    info = sf.info(str(file_path))
    return {
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "size_mb": os.path.getsize(file_path) / (1024 * 1024)
    }



def show_waveform(file_path, title):
    audio, sample_rate = sf.read(str(file_path))

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Keep the graph fast, even for longer files.
    step = max(1, len(audio) // 5000)
    audio = audio[::step]

    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.plot(np.linspace(0, 1, len(audio)), audio, linewidth=0.6)

    ax.set_title(title, color="white")
    ax.set_xlabel("Relative audio timeline", color="white")
    ax.set_ylabel("Amplitude", color="white")
    ax.set_facecolor("#191d32")
    fig.patch.set_facecolor("#191d32")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#555570")

    st.pyplot(fig)
    plt.close(fig)




# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 🎵 MusicEnhance AI")
    st.caption("SMART AUDIO STUDIO")

    st.markdown("---")

    st.markdown("### About the Project")
    st.write(
        "An academic demonstration of machine learning "
        "for music source separation."
    )

    st.markdown("### Technology Stack")
    st.markdown(
        "- Python\n"
        "- Streamlit\n"
        "- Pretrained Demucs model\n"
        "- NumPy and SoundFile\n"
        "- Matplotlib"
    )

    st.markdown("---")
    st.caption("Seventh Semester | Minor Project")


# ---------------- HEADER ----------------


st.markdown("""
<div style="padding: 30px; border-radius: 18px;
background: linear-gradient(120deg, #312e81, #6d28d9, #0f766e);
margin-bottom: 24px;">

<div style="font-size: 13px; letter-spacing: 3px; color: #ddd6fe;">
AI AUDIO LAB
</div>

<h1 style="color: white; margin-bottom: 8px;">
MusicEnhance AI 🎵
</h1>

<p style="color: #ede9fe; font-size: 17px;">
Explore your music. Separate the sound.
</p>

<p style="color: #ddd6fe;">
Machine-learning-based music source separation
</p>

</div>
""", unsafe_allow_html=True)



st.markdown("## Your Audio Workspace")

st.write(
    "Upload a music track, run the pretrained AI model, "
    "and listen to its individual musical components."
)

st.caption(
    "Supported formats: WAV, MP3, FLAC and OGG. "
    "Short audio clips are recommended for testing."
)

st.markdown("### How It Works")

step1, step2, step3 = st.columns(3)

with step1:
    st.markdown("#### 01 · Upload")
    st.write("Choose your audio file.")

with step2:
    st.markdown("#### 02 · Process")
    st.write("Run the pretrained AI model.")

with step3:
    st.markdown("#### 03 · Explore")
    st.write("Listen to and download the results.")



# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload your music",
    type=["wav", "mp3", "flac", "ogg"],
    help="For your first test, use a short audio file."
)

if uploaded_file is not None:
    st.success(f"Selected file: {uploaded_file.name}")

    st.audio(uploaded_file.getvalue())

    if st.button("✨ Process Music with AI"):
        try:
            with st.spinner(
                "Running the pretrained AI model. "
                "This may take several minutes..."
            ):
                # Each run gets its own working directory.
                work_dir = Path(tempfile.mkdtemp(prefix="music_enhance_"))
                input_path = work_dir / ("input_audio" + Path(
                    uploaded_file.name
                ).suffix.lower())

                input_path.write_bytes(uploaded_file.getvalue())

                # Check that the uploaded audio can be read.
                original_info = get_audio_info(input_path)

                output_dir = work_dir / "outputs"
                output_dir.mkdir()

                # Run the pretrained Demucs source separation model.
                command = [
                    sys.executable,
                    "-m", "demucs",
                    "-n", "htdemucs",
                    "--out", str(output_dir),
                    str(input_path)
                ]

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=900
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        result.stderr[-4000:]
                        or result.stdout[-4000:]
                        or "The AI model could not process this file."
                    )

                stem_dir = output_dir / "htdemucs" / "input_audio"

                stems = {}
                for stem_name in [
                    "vocals", "drums", "bass", "other"
                ]:
                    stem_path = stem_dir / f"{stem_name}.wav"
                    if stem_path.exists():
                        stems[stem_name] = str(stem_path)

                if not stems:
                    raise RuntimeError(
                        "The model completed, but no output stems were found."
                    )

                st.session_state["music_result"] = {
                    "original": str(input_path),
                    "info": original_info,
                    "stems": stems,
                    "work_dir": str(work_dir)
                }

            st.success("AI processing completed!")

        except subprocess.TimeoutExpired:
            st.error(
                "Processing took longer than 15 minutes. "
                "Try a shorter audio file or use a computer with a GPU."
            )

        except Exception as error:
            st.error(f"Processing failed: {error}")
            st.caption(
                "Check the terminal for installation errors. "
                "Make sure Demucs and PyTorch installed successfully."
            )


# ---------------- RESULTS ----------------

if "music_result" in st.session_state:
    result = st.session_state["music_result"]

    st.markdown("---")
    st.header("📊 Audio Analysis")

    info = result["info"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Duration", f"{info['duration']:.1f} sec")
    col2.metric("Sample Rate", f"{info['sample_rate']:,} Hz")
    col3.metric("Channels", info["channels"])
    col4.metric("File Size", f"{info['size_mb']:.2f} MB")

    st.subheader("Original Audio")
    st.audio(Path(result["original"]).read_bytes())

    with st.expander("View original waveform"):
        show_waveform(result["original"], "Original Audio Waveform")

    st.markdown("---")
    st.header("🎚️ AI-Separated Music")

    st.write(
        "The model estimates four components of the original mix. "
        "Listen to each component below."
    )

    descriptions = {
        "vocals": "Singing and vocal content",
        "drums": "Percussion and drum sounds",
        "bass": "Bass instruments",
        "other": "Other musical accompaniment"
    }

    for stem_name, stem_path in result["stems"].items():
        st.subheader(stem_name.capitalize())
        st.caption(descriptions.get(stem_name, "Separated audio"))
        st.audio(stem_path)

        with open(stem_path, "rb") as audio_file:
            st.download_button(
                label=f"⬇️ Download {stem_name.capitalize()}",
                data=audio_file.read(),
                file_name=f"{stem_name}.wav",
                mime="audio/wav",
                key=f"download_{stem_name}"
            )

    st.markdown("---")
    st.caption(
        "MusicEnhance AI | Academic demonstration of pretrained "
        "machine-learning-based audio source separation."
    )

