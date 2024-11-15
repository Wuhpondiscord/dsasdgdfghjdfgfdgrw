import streamlit as st
import torch
from transformers import (
    pipeline, 
    GPTNeoForCausalLM, 
    GPT2Tokenizer, 
    Pipeline  
)
from diffusers import StableDiffusionPipeline
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: {device}")

@st.cache_resource
def load_text_model(model_name, device):
    try:
        if model_name == "GPT-Neo 125M":
            tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
            model = GPTNeoForCausalLM.from_pretrained(
                "EleutherAI/gpt-neo-125M",
                low_cpu_mem_usage=True
            ).to(device)
        elif model_name == "GPT-2":
            model = pipeline(
                "text-generation",
                model="gpt2",
                device=0 if device == "cuda" else -1
            )
            tokenizer = None
        else:
            st.error("Selected text model is not supported.")
            return None, None
        return tokenizer, model
    except OSError as e:
        st.error(f"Failed to load {model_name}: {e}")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading {model_name}: {e}")
        return None, None

@st.cache_resource
def load_image_model(model_name, device):
    try:
        if model_name == "Stable Diffusion v1-4":
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=dtype
            )
        elif model_name == "Stable Diffusion v2-1":
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=dtype
            )
        else:
            st.error("Selected image model is not supported.")
            return None
        pipe = pipe.to(device)
        pipe.safety_checker = None
        return pipe
    except OSError as e:
        st.error(f"Failed to load {model_name}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading {model_name}: {e}")
        return None

def generate_text(prompt, tokenizer, model, device, max_length=50):
    if isinstance(model, Pipeline): 
        generated = model(prompt, max_length=max_length, do_sample=True, temperature=0.7, num_return_sequences=1)
        return generated[0]['generated_text']
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            num_beams=2,
            early_stopping=True
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

def generate_image(prompt, pipe, num_inference_steps=25):
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
    return image

st.title("Divine Image Generator")
st.write("This application channels divine inspiration to create religious-themed images.")
st.sidebar.header("Model Selection")
text_models = ["GPT-Neo 125M", "GPT-2"]  
selected_text_model = st.sidebar.selectbox("Select Text Model", text_models)
image_models = ["Stable Diffusion v1-4", "Stable Diffusion v2-1"]
selected_image_model = st.sidebar.selectbox("Select Image Model", image_models)
with st.spinner("Loading models..."):
    tokenizer, text_model = load_text_model(selected_text_model, device)
    image_pipe = load_image_model(selected_image_model, device)
if selected_text_model != "GPT-2" and tokenizer is None or text_model is None or image_pipe is None:
    st.stop()
if selected_text_model == "GPT-2" and text_model is None or image_pipe is None:
    st.stop()
user_input = st.text_input("Enter a theme or concept for divine inspiration (e.g., 'divine guidance'):")
if st.button("Generate Divine Image"):
    if user_input:
        with st.spinner("Connecting with the divine..."):
            prompt = f"As God, speak to the people with wisdom and divine inspiration about: {user_input}"
            if selected_text_model == "GPT-2":
                divine_prompt = generate_text(prompt, tokenizer, text_model, device)
            else:
                divine_prompt = generate_text(prompt, tokenizer, text_model, device)
            st.write("**Divine Message:**", divine_prompt)
            st.write("**Generating image...**")
            divine_image = generate_image(divine_prompt, image_pipe)
            st.image(divine_image, caption="Generated Religious Image", use_container_width=True)
    else:
        st.warning("Please enter a theme or concept for divine inspiration.")
