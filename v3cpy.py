import streamlit as st
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# -------------------------
# Load API key
# -------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------------
# Nano Banana Pro model
# -------------------------
VISION_MODEL = "gemini-3-pro-image-preview"

# -------------------------
# Generate image function
# -------------------------
def generate_image_with_reference(
    lehenga_img: Image.Image,
    closeup_img: Image.Image | None = None,
    blouse_img: Image.Image | None = None
) -> bytes | None:
    """
    Generates a 2048x2048 image from the uploaded lehenga + optional references.
    """
    contents = [
        lehenga_img,
        "Generate a photorealistic image of a model wearing this lehenga. "
        "Do NOT change color, embroidery, motifs, pleats, or fabric. "
        "Preserve every design detail exactly. Studio lighting, head-to-knee, 2048x2048."
    ]

    if closeup_img:
        contents.append(closeup_img)
        contents.append("Reference close-up: preserve embroidery and fabric texture exactly.")

    if blouse_img:
        contents.append(blouse_img)
        contents.append("Reference blouse: preserve cut, sleeve, and stitch design exactly.")

    try:
        response = client.models.generate_content(
            model=VISION_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(
                    image_size="2K",
                    aspect_ratio="1:1"
                ),
                response_modalities=["IMAGE"]
            )
        )

        part = response.parts[0]
        img = part.as_image()
        buf = BytesIO()
        img.save(buf, "JPEG", quality=95)
        buf.seek(0)
        return buf.read()

    except Exception as e:
        st.error(f"Failed to generate image: {e}")
        return None

# -------------------------
# Streamlit UI
# -------------------------
st.title("👗 Lehenga Try-On — Nano Banana Pro 2K (Fixed)")

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
        with st.spinner("Generating image with reference..."):
            img_bytes = generate_image_with_reference(lehenga_img, closeup_img, blouse_img)
            if img_bytes:
                out = Image.open(BytesIO(img_bytes)).convert("RGB")
                st.subheader("Generated Image (2048×2048)")
                st.image(out, use_column_width=True)
                buf = BytesIO()
                out.save(buf, "JPEG", quality=95)
                buf.seek(0)
                st.download_button(
                    "📥 Download",
                    data=buf,
                    file_name="lehenga_tryon.jpg",
                    mime="image/jpeg"
                )
            else:
                st.error("Image generation failed — check model availability or API quota.")
