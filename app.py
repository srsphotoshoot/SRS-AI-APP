import streamlit as st
from PIL import Image
import google.generativeai as genai
import base64
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY= os.getenv("GOOGLE_API_KEY")


genai.configure(api_key=GEMINI_API_KEY)

def pick_models():
    models = genai.list_models()
    # flatten to names
    names = [m.name for m in models]
    # candidate for text/prompt generation
    # prefer a multimodal text model that supports images+text → text
    text_model = None
    for nm in names:
        if nm.startswith("models/gemini-2.5-flash") or nm.startswith("models/gemini-2.0-flash"):
            text_model = nm
            break
    if text_model is None:
        # fallback to generic text-only
        text_model = names[0] if names else None

    # candidate for image generation
    image_model = None
    for nm in names:
        # pick a model that supports image output
        if "image" in nm.lower() or nm.endswith("-image-preview") or nm.endswith("-image"):
            image_model = nm
            break
    return text_model, image_model

TEXT_MODEL, IMAGE_MODEL = pick_models()
if not TEXT_MODEL:
    raise RuntimeError("No valid Gemini model found for text generation — check your API access.")
st.write("Using prompt‑generation model:", TEXT_MODEL)
if IMAGE_MODEL:
    st.write("Using image‑generation model:", IMAGE_MODEL)
else:
    st.write("⚠️ No image‑generation model available — image output will be skipped.")

# ====== Functions ======
def generate_prompt(img: Image.Image) -> str:
    PROMPT_TEMPLATE = """
    Analyze the uploaded image and describe the lehenga in maximum detail including:
    • embroidery & stone work
    • blouse pattern
    • dupatta design
    • skirt border & flare
    • colour & fabric texture

    Then convert the description into an instruction in this exact format:

    "Generate an image of a model wearing the exact same lehenga from the image,
    preserving embroidery, pattern layout, colour tone, blouse design and dupatta style.
    Increase realism, fabric texture, and detail. No design changes."

    Output should be ONLY the final instruction prompt (no explanation).
    """
    model = genai.GenerativeModel(TEXT_MODEL)
    result = model.generate_content([PROMPT_TEMPLATE, img])
    return result.text

def generate_image_from_prompt(prompt: str) -> bytes | None:
    if not IMAGE_MODEL:
        return None
    model = genai.GenerativeModel(IMAGE_MODEL)
    result = model.generate_content(prompt, stream=False)
    # first candidate, inline_data assumed
    b64 = result.candidates[0].content.parts[0].inline_data.data
    return base64.b64decode(b64)

# ====== Streamlit UI ======
st.title("👗 Lehenga Try‑On Generator (Gemini — dynamic model select)")

uploaded_file = st.file_uploader("Upload Lehenga Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    input_img = Image.open(uploaded_file)
    st.image(input_img, caption="Uploaded Lehenga", width=350)

    if st.button("Generate Model Image"):
        with st.spinner("Generating prompt from image..."):
            prompt = generate_prompt(input_img)
        st.subheader("Generated Prompt")
        st.write(prompt)

        if IMAGE_MODEL:
            with st.spinner("Generating model wearing lehenga..."):
                output_bytes = generate_image_from_prompt(prompt)
            if output_bytes:
                st.subheader("Final Generated Image")
                st.image(output_bytes, use_column_width=True)
                st.download_button(
                    label="📥 Download Image",
                    data=output_bytes,
                    file_name="model_lehenga.jpg",
                    mime="image/jpeg"
                )
            else:
                st.error("Image generation failed — model returned no data.")
        else:
            st.info("No image‑generation model available — only prompt generated.")
