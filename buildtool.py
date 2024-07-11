# text to Image

import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Model Path/Repo Information
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" 

# Load model (Executed only once for efficiency)
@st.cache_resource
def load_sdxl_pipeline():
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
    #pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return pipe

# Streamlit UI
st.title("Image Generation")
prompt = st.text_input("Enter your image prompt:")

if st.button("Generate Image"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        pipe = load_sdxl_pipeline()  # Load the pipeline from cache
        with torch.no_grad():
            image = pipe(prompt).images[0] 

        st.image(image)
