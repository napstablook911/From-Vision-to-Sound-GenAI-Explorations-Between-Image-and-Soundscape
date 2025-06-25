import sys
sys.setrecursionlimit(2000) # Or a higher value, e.g., 3000, 5000.
import streamlit as st
from PIL import Image
import io
import soundfile as sf
import numpy as np
import torch
from transformers import pipeline, AutoFeatureExtractor
from diffusers import StableAudioPipeline
from diffusers.pipelines.stable_audio import StableAudioPipeline # Corrected import path

# --- Configuration ---
# Determine the optimal device for model inference
# Prioritize CUDA (NVIDIA GPUs), then MPS (Apple Silicon), fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Use float16 for reduced memory and faster inference on compatible hardware (GPU/MPS)
# Fallback to float32 for CPU for better stability
TORCH_DTYPE = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32

# --- Cached Model Loading Functions ---
@st.cache_resource(show_spinner="Loading Image Captioning Model (BLIP)...")
def load_blip_model():
    """
    Loads the BLIP image captioning model using Hugging Face transformers pipeline.
    The model is cached to prevent reloading on every Streamlit rerun.
    """
    try:
        captioner = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            torch_dtype=TORCH_DTYPE,
            device=0 if DEVICE == "cuda" else -1  # 0 for GPU, -1 for CPU/MPS
        )
        return captioner
    except Exception as e:
        st.error(f"Failed to load BLIP model: {e}")
        return None


@st.cache_resource(show_spinner="Loading Audio Generation Model (Stable Audio Open Small)...")
def load_stable_audio_model():
    """
    Loads the Stable Audio Open Small pipeline using Hugging Face diffusers.
    The pipeline is cached to prevent reloading on every Streamlit rerun.
    """
    try:
        from diffusers import DDIMScheduler

        # Use the StableAudioPipeline imported from the specific submodule
        pipeline = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0",
            torch_dtype=TORCH_DTYPE
        )

        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(DEVICE)
        return pipeline
    except Exception as e:
        st.error(f"Failed to load Stable Audio model: {e}")
        return None


# --- Audio Conversion Utility ---
def convert_numpy_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """
    Converts a NumPy audio array to an in-memory WAV byte stream.
    This avoids writing temporary files to disk, which is efficient and
    suitable for ephemeral environments like Hugging Face Spaces.
    Expected input `audio_array` shape: (batch_size, channels, frames) if coming from pipeline
    """
    byte_io = io.BytesIO()
    
    # FIX: Remove batch dimension and ensure correct shape for soundfile
    if audio_array.ndim == 3 and audio_array.shape[0] == 1:
        audio_array = audio_array[0] # Remove batch dimension: now (channels, frames)

    # soundfile typically expects (frames, channels) for stereo.
    # If it's (channels, frames), transpose it.
    if audio_array.ndim == 2 and audio_array.shape[0] == 2: # Stereo: (channels, frames)
        audio_array = audio_array.T # Transpose to (frames, channels)
    elif audio_array.ndim == 1: # Mono: (frames,)
        pass # Already in correct shape

    # Ensure the audio_array has a consistent data type before writing
    audio_array = audio_array.astype(np.float32)

    # Write the NumPy array to the in-memory BytesIO object as a WAV file
    sf.write(byte_io, audio_array, sample_rate, format='WAV', subtype='FLOAT')
    
    # IMPORTANT: Reset the stream position to the beginning before reading
    byte_io.seek(0) 
    return byte_io.read()

# --- Streamlit App Layout ---
st.set_page_config(layout="centered", page_title="Image-to-Soundscape Generator")
st.title("🏞️ Image-to-Soundscape Generator 🎶")
st.markdown("Upload a landscape image, and let AI transform it into a unique soundscape!")

# Initialize session state for persistence across reruns
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False
if "generated_caption_for_edit" not in st.session_state:
    st.session_state.generated_caption_for_edit = ""

# --- UI Components ---
uploaded_file = st.file_uploader("Choose a landscape image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.session_state.image_uploaded = True
    image = Image.open(uploaded_file).convert("RGB") # Ensure image is in RGB format
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Button to trigger the caption generation
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            try:
                captioner = load_blip_model()
                if captioner is None:
                    st.error("Image captioning model could not be loaded. Please try again.")
                    st.session_state.image_uploaded = False
                    st.stop()
                st.toast("BLIP model loaded successfully.")

                caption_results = captioner(image)
                generated_caption = caption_results[0]['generated_text']
                st.session_state.generated_caption_for_edit = generated_caption
                st.success("Caption generated!")
            except Exception as e:
                st.error(f"An error occurred during caption generation: {e}")
                st.exception(e)

    # Display and allow editing of the generated caption if available
    if st.session_state.generated_caption_for_edit:
        st.subheader("Edit Soundscape Prompt:")
        # Allow user to edit the prompt
        soundscape_prompt_input = st.text_area(
            "Based on the image caption, feel free to refine the prompt for the soundscape. "
            "Add descriptive words for sounds (e.g., 'gentle breeze', 'birds chirping', 'water flowing').",
            value=f"A soundscape of {st.session_state.generated_caption_for_edit}",
            height=100
        )
        st.info(f"Final prompt for audio: '{soundscape_prompt_input}'")

        # Button to trigger the audio generation pipeline
        if st.button("Generate Soundscape from Prompt"):
            st.session_state.audio_bytes = None # Clear previous audio
            
            with st.spinner("Generating soundscape... This may take a moment."):
                try:
                    # 2. Load Stable Audio model and generate audio
                    audio_pipeline = load_stable_audio_model()
                    if audio_pipeline is None:
                        st.error("Audio generation model could not be loaded. Please try again.")
                        st.session_state.image_uploaded = False # Reset to allow re-upload
                        st.stop()
                    st.toast("Stable Audio model loaded successfully.")

                    # Generate audio with optimized parameters for speed
                    audio_output = audio_pipeline(
                        prompt=soundscape_prompt_input, # Use the potentially edited prompt
                        num_inference_steps=100,   # Increased for better quality
                        audio_end_in_s=5,         # 5 seconds audio length (within 11s limit for small model)
                        negative_prompt="low quality, average quality, distorted",
                        guidance_scale=9.0        # Increased for better prompt adherence
                    )
                    
                    # Convert PyTorch Tensor to NumPy array before passing to convert_numpy_to_wav_bytes
                    audio_numpy_array = audio_output.audios.cpu().numpy()
                    sample_rate = audio_pipeline.vae.sampling_rate 

                    st.info(f"Audio numpy array shape: {audio_numpy_array.shape}, dtype: {audio_numpy_array.dtype}")
                    st.info(f"Sample rate: {sample_rate}")


                    # 3. Convert NumPy array to WAV bytes and store in session state
                    st.session_state.audio_bytes = convert_numpy_to_wav_bytes(audio_numpy_array, sample_rate)
                    
                    st.success("Soundscape generated successfully!")
                    st.info(f"Length of generated WAV bytes: {len(st.session_state.audio_bytes)} bytes")


                except Exception as e:
                    st.error(f"An error occurred during generation: {e}")
                    st.session_state.audio_bytes = None # Clear any partial audio
                    st.session_state.image_uploaded = False # Reset to allow re-upload
                    st.exception(e) # Display full traceback for debugging 

# Display generated soundscape if available in session state
if st.session_state.audio_bytes:
    st.subheader("Generated Soundscape:")
    st.audio(st.session_state.audio_bytes, format='audio/wav')
    st.markdown("You can download the audio using the controls above.")

    st.download_button(
        label="⬇️ Download Soundscape",
        data=st.session_state.audio_bytes,
        file_name="soundscape.wav",
        mime="audio/wav"
    )


# Reset button for new image upload
if st.session_state.image_uploaded and st.button("Upload New Image"):
    st.session_state.audio_bytes = None
    st.session_state.image_uploaded = False
    st.session_state.generated_caption_for_edit = "" # Clear stored caption
    st.rerun() # Rerun the app to clear the file uploader
