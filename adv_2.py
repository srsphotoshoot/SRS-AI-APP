import streamlit as st
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# -------------------------
# Load API key  (working)
# ----------------- --------
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------------
# Nano Banana Pro model
# -------------------------
VISION_MODEL = "gemini-3-pro-image-preview"

# -------------------------
# Prompt generation function
# -------------------------
def generate_prompt(lehenga_img: Image.Image, closeup_img: Image.Image | None, blouse_img: Image.Image | None) -> str:
    """
    Generates a strict single-line prompt instruction for Nano Banana Pro
    based on the provided reference images.
    """
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
    # Generate prompt text (Nano Banana Pro will interpret)
    response = client.models.generate_content(
        model=VISION_MODEL,
        contents=[PROMPT_TEMPLATE]
    )
    return response.parts[0].text.strip()

# -------------------------
# Generate image from prompt
# -------------------------
def generate_image_from_prompt(prompt_instruction: str) -> bytes | None:
    """
    Generates a 2048x2048 image from a single-line prompt using Nano Banana Pro.
    Returns raw bytes suitable for PIL or download.
    """
    try:
        response = client.models.generate_content(
            model=VISION_MODEL,
            contents=[prompt_instruction],
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(
                    image_size="2K",
                    aspect_ratio="1:1"
                ),
                response_modalities=["IMAGE"]
            )
        )

        # Check if .as_image() returns PIL.Image or bytes
        part = response.parts[0]
        try:
            # Try as PIL Image
            img = part.as_image()
            buf = BytesIO()
            img.save(buf, "JPEG", quality=95)  # Correct usage: no "format=" keyword
            buf.seek(0)
            return buf.read()
        except TypeError:
            # Already bytes
            return part.inline_data.data  # or part.as_bytes()
            
    except Exception as e:
        st.error(f"Failed to generate image: {e}")
        return None


# -------------------------
# Streamlit UI
# -------------------------
st.title("👗 Lehenga Try-On — Nano Banana Pro 2K")

lehenga_file = st.file_uploader("Upload Lehenga Image (full view)", type=["jpg","jpeg","png"])
closeup_file = st.file_uploader("Upload Design Close-up (optional)", type=["jpg","jpeg","png"])
blouse_file  = st.file_uploader("Upload Blouse Reference (optional)", type=["jpg","jpeg","png"])

lehenga_img = Image.open(lehenga_file).convert("RGB") if lehenga_file else None
closeup_img = Image.open(closeup_file).convert("RGB") if closeup_file else None
blouse_img  = Image.open(blouse_file).convert("RGB") if blouse_file else None

# Preview uploaded images
if lehenga_img:
    st.image(lehenga_img, caption="Lehenga (full view)", width=360)
if closeup_img:
    st.image(closeup_img, caption="Design Close-up", width=240)
if blouse_img:
    st.image(blouse_img, caption="Blouse Reference", width=240)

# Generate button
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
                st.error("Image generation failed — check model availability or API quota.")
