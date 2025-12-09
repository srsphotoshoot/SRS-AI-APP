import streamlit as st
from PIL import Image
import google.generativeai as genai
import base64
from dotenv import load_dotenv
import os
from io import BytesIO

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Use a supported image-generation model
VISION_MODEL = "gemini-2.5-flash-image"  # for multi-image grounding + image output
# (If you later want a pure text→image mode, you could use an Imagen model instead.)

def generate_prompt(lehenga_img: Image.Image, closeup_img: Image.Image | None, blouse_img: Image.Image | None):
    PROMPT_TEMPLATE = """
You are a professional fashion photographer and textile conservator. Use the provided reference images and create a strict instruction (single-line only, no explanations) for image generation:

References:
- Lehenga full-view (silhouette, colour distribution)
- Close-up (embroidery, fabric texture, stonework, borders) — if provided
- Blouse reference (blouse cut, neckline, sleeve, stitch design) — if provided

Rules (MUST follow exactly):
- Generate a photorealistic image of a model wearing the *exact same lehenga* from references.
- Preserve blouse, embroidery, stones, borders, fabric texture, colour, motifs placement, pleats/flare, dupatta drape.
- Do NOT change or remove any design element.
- Compose the image to focus tightly on the lehenga (head-to-knee), highlighting detail and texture.
- Output image size 2048×2048 (2 K), high quality, studio lighting, no added props, no logos or text, no background distractions.

Output only one line: starting with “Generate a 2048x2048 photorealistic image of a model wearing…” and including constraints as above.
    """
    inputs = [PROMPT_TEMPLATE, lehenga_img]
    if closeup_img:
        inputs.append(closeup_img)
    if blouse_img:
        inputs.append(blouse_img)

    model = genai.GenerativeModel(VISION_MODEL)
    result = model.generate_content(inputs)
    return result.text.strip()

def generate_image_from_prompt(prompt_instruction: str):
    model = genai.GenerativeModel(VISION_MODEL)
    result = model.generate_content(prompt_instruction, stream=False, response_modalities=['Image'])
    try:
        b64 = result.candidates[0].content.parts[0].inline_data.data
    except Exception as e:
        st.error(f"Failed to parse image data: {e}")
        return None
    return base64.b64decode(b64)

st.title("👗 Lehenga Try-On — High Detail 2K")

lehenga_file = st.file_uploader("Upload Lehenga Image (full view)", type=["jpg","jpeg","png"])
closeup_file = st.file_uploader("Upload Design Close-up (embroidery/stones) — optional", type=["jpg","jpeg","png"])
blouse_file  = st.file_uploader("Upload Blouse Reference — optional", type=["jpg","jpeg","png"])

if lehenga_file:
    lehenga_img = Image.open(lehenga_file).convert("RGB")
    st.image(lehenga_img, caption="Lehenga (full view)", width=360)
else:
    lehenga_img = None

closeup_img = None
if closeup_file:
    closeup_img = Image.open(closeup_file).convert("RGB")
    st.image(closeup_img, caption="Design Close-up", width=240)

blouse_img = None
if blouse_file:
    blouse_img = Image.open(blouse_file).convert("RGB")
    st.image(blouse_img, caption="Blouse Reference", width=240)

if st.button("Generate 2K Try-On"):
    if not lehenga_img:
        st.error("Please upload the full-view lehenga image.")
    else:
        with st.spinner("Generating prompt..."):
            prompt = generate_prompt(lehenga_img, closeup_img, blouse_img)
            st.subheader("Generated Prompt")
            st.write(prompt)
        with st.spinner("Generating image..."):
            img_bytes = generate_image_from_prompt(prompt)
            if img_bytes:
                out = Image.open(BytesIO(img_bytes)).convert("RGB")
                st.subheader("Generated Image (2048×2048)")
                st.image(out, use_column_width=True)
                buf = BytesIO()
                out.save(buf, format="JPEG", quality=95)
                buf.seek(0)
                st.download_button("📥 Download", data=buf, file_name="lehenga_tryon.jpg", mime="image/jpeg")
            else:
                st.error("Image generation failed — check model availability or quota.")
