# From-Vision-to-Sound: GenAI Explorations Between Image and Soundscape

## Project Overview

This project explores the potential of GenAI in translating visual content into immersive soundscapes. Starting with landscape images, the AI first generates a textual description using an image captioning model, and then transforms that description into a simulated auditory environment using text-to-audio synthesis.

Sight has always played a key role in human survival. In modern times, it has taken a predominant place in the development of technologies, leading to the assumption that the other senses are less important.

Hearing is often overlooked, despite its vital role today in the entertainment industry. This presence is mostly associated with visual inputs—images, videos, or more general aesthetics that help make sounds recognizable to audiences.

The goal of this project is to investigate how deep learning models interpret visual input and how this interpretation manifests acoustically, offering insights into how visual stimuli can influence auditory imagination in AI systems.

---

## Objectives

* Investigate the relationship between visual and auditory perception.
* Assess the semantic and sensory coherence between images and the corresponding soundscapes.
* Create a user-friendly web interface to explore these connections interactively.

---

## Tools and Technologies

* **Python + AI Libraries**:

  * `transformers`, `diffusers`, `torch`, `numpy`, `soundfile`, `Pillow`, `torchsde`, `accelerate`

* **Image Captioning**:

  * [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base) via Hugging Face `transformers`

* **Audio Generation**:

  * [`stabilityai/stable-audio-open-1.0`](https://huggingface.co/stabilityai/stable-audio-open-1.0) via Hugging Face `diffusers`

* **Web Interface**:

  * Built using `Streamlit` for interaction and visualization

* **Development Environment**:

  * Local machine with GPU acceleration

---

## AI Techniques Employed

* **Vision-to-Text**: Uses the BLIP (Bootstrapped Language Image Pretraining) model to produce image captions from landscape inputs.
* **Text-to-Audio**: Uses Stability AI’s *Stable Audio Open 1.0* to synthesize a brief audio scene from descriptive captions.

---

## Expected Outcomes

* **Soundscapes**: Original 5-second soundscapes generated from visual inputs.
* **Interactive Interface**: A Streamlit web app to upload an image, generate a caption, and listen to the resulting sound.
* **Soundscapes folder**: [Google drive folder](https://drive.google.com/drive/folders/1C4PcwewEqcHzgb4Ak4kEKICSO8oAyHnF?usp=sharing) with results organized by prompt, available for consultation. 

---

# How to install

Local Installation Guide (macOS). For now only local installation with macOS system is available.

This guide will help you set up and run the **Soundscape Generator** locally on your Mac.

### Requirements

* macOS with Python 3.9 or newer installed
* `pip` (Python package installer)
* A terminal application (like Terminal or iTerm)

### Project Files

Make sure you have the following files in a single folder:

* `app.py` — the main application script
* `requirements.txt` — the list of required Python packages

---

### Installation Steps

1. **Open Terminal** and navigate to the folder where your files are located.
   Example:

   ```bash
   cd path/to/your/folder
   ```

2. **(Optional but recommended)**: Create and activate a virtual environment

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies** from `requirements.txt`

   ```bash
   pip install -r requirements.txt
   ```


4. **To run the Streamlit app locally:**

  ```bash
   streamlit run app.py
   ```

Make sure you have an internet connection for the first run (models will be downloaded from Hugging Face).

---


## Performance Notes

> A **GPU is strongly recommended** for stable and fast performance.

This app uses high-performance transformer and diffusion models. Performance varies significantly by hardware.

If you're using a Mac with M1/M2 chip, make sure your `torch` installation supports MPS:

```python
import torch
print(torch.backends.mps.is_available())
```

Perfetto, ecco una versione migliorata e ben strutturata della sezione **Example Workflow**, con spazi per le immagini che raffigurano i passaggi. È scritta in uno stile chiaro e professionale, adatto a un README o a una presentazione di progetto.

---

## Example

This example outlines how the application works in a few intuitive steps:

### 1. **Upload an Image**

Start by uploading a landscape image (e.g., a forest, beach, or mountain scene).

![Website image](https://github.com/napstablook911/From-Vision-to-Sound-GenAI-Explorations-Between-Image-and-Soundscape/blob/main/images/Screenshot%202025-07-04%20at%2010.26.33.png)

---

### 2. **Generate a Caption**

The app uses a vision-language model (BLIP) to generate a descriptive caption for the uploaded image.

---

### 3. **Edit the Caption (Optional)**

You can refine the generated caption by adding auditory details (e.g., "gentle wind", "waves crashing", "birds singing"). This edited text will be used as the soundscape generation prompt.

![Website image](https://github.com/napstablook911/From-Vision-to-Sound-GenAI-Explorations-Between-Image-and-Soundscape/blob/main/images/Screenshot%202025-07-04%20at%2010.27.06.png)

---

### 4. **Generate the Soundscape**

Once the prompt is ready, click the button to generate a unique soundscape using Stable Audio Open 1.0.

---

### 5. **Listen & Download**

Play back the generated audio directly in the browser, and optionally download it as a `.wav` file.

![Website image](https://github.com/napstablook911/From-Vision-to-Sound-GenAI-Explorations-Between-Image-and-Soundscape/blob/main/images/Screenshot%202025-07-04%20at%2010.29.23.png)

---

### Troubleshooting

* If you see a `Permission denied` or `zsh: command not found`, try:

  ```bash
  chmod +x app.py
  ```

* If a package fails to install, make sure you're using the correct Python version and that your virtual environment is activated.



 

