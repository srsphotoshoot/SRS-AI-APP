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


VISION_MODEL = "models/gemini-1.0-basic"
IMAGE_MODEL  = "models/gemini-2.5-flash-image"


def pil_to_bytes(img: Image.Image, fmt="PNG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()

def generate_prompt(lehenga_img: Image.Image, closeup_img: Image.Image | None, blouse_img: Image.Image | None):
    """
    Create a strict, multi-image grounded prompt that instructs Gemini
    to produce a 2K realistic image of a model wearing the exact same lehenga.
    """
 
    PROMPT_TEMPLATE = """
You are an expert fashion photographer and textile conservator with vision-to-image capabilities.
Carefully analyze the reference images provided and produce a single-line generation instruction (only the instruction — no explanations).

References:
1) Lehenga full-view image: shows overall silhouette, volume, and colour distribution.
2) Close-up design image: shows embroidery, stones, thread patterns, borders, textures (if provided).
3) Blouse reference image: shows blouse cut, stitch pattern, neckline, sleeve style (if provided).

Required behavior (MUST follow exactly):
- Recreate the **lehenga exactly as in the references**: identical embroidery placement, stonework, motifs, borders, pleat/flare, colour tone, fabric texture, and blouse design. Do not alter any design element.
- Preserve fabric micro-texture, stitch direction, and motif scale. If the close-up shows a repeated motif, maintain the same repeat spacing on the skirt.
- Output image must be photorealistic, model wearing the lehenga, full body (head to knee), centered and tightly focused on the lehenga so DETAILS are visible.
- Use natural studio lighting highlighting fabric texture; no heavy retouching, no added accessories that occlude the lehenga.
- Do NOT change colour, trim, embroidery, border, or blouse shape. No additional patterns or embellishments. No cropping that removes details.
- Image size: request 2048x2048 (2K) final output.

Negative constraints (things to avoid):
- Do NOT redesign, recolor, or simplify any patterns.
- Do NOT add or remove stones, patches, or borders.
- Do NOT change blouse cut or dupatta draping style.
- Do NOT include text, logos, watermarks, or extra props that hide the dress.

Now produce a single concise generation instruction (one line) that a generative image model can use to create the final 2048x2048 photorealistic image. Use the words: "Generate a 2048x2048 photorealistic image of a model wearing the exact same lehenga as the references, preserving all listed details." Then append minimal clarifying constraints about zoom and focus.
    """

    # Prepare input list: prompt + image(s)
    inputs = [PROMPT_TEMPLATE, lehenga_img]
    if closeup_img:
        inputs.append(closeup_img)
    if blouse_img:
        inputs.append(blouse_img)

    model = genai.GenerativeModel(VISION_MODEL)

    result = model.generate_content(inputs)
  
    return result.text.strip()

def generate_image_from_prompt(prompt_instruction: str):
    """
    Call the image model with the single-line instruction and request 2048x2048 output.
    """
    model = genai.GenerativeModel(IMAGE_MODEL)

    result = model.generate_content(prompt_instruction, stream=False)

    try:
        image_base64 = result.candidates[0].content.parts[0].inline_data.data
    except Exception as e:
        st.error(f"Image generation response parsing failed: {e}")
        return None
    image_bytes = base64.b64decode(image_base64)
    return image_bytes


st.title("👗 Lehenga Try-On — High-Detail 2K Generator")

st.markdown(
    """
Upload:
- A full-view **lehenga image** (required).
- A **close-up** of embroidery/stonework (optional but strongly recommended).
- A **blouse** reference (optional but recommended for exact blouse reproduction).
"""
)

lehenga_file = st.file_uploader("Upload Lehenga Image (Full view)", type=["jpg","jpeg","png"])
closeup_file = st.file_uploader("Upload Close-up Design (embroidery/stitch) — A", type=["jpg","jpeg","png"])
blouse_file  = st.file_uploader("Upload Blouse Reference — B", type=["jpg","jpeg","png"])

if lehenga_file:
    try:
        lehenga_img = Image.open(lehenga_file).convert("RGB")
        st.image(lehenga_img, caption="Lehenga (full view)", width=360)
    except Exception as e:
        st.error(f"Failed to open lehenga image: {e}")
        lehenga_img = None
else:
    lehenga_img = None

if closeup_file:
    try:
        closeup_img = Image.open(closeup_file).convert("RGB")
        st.image(closeup_img, caption="Design Close-up (A)", width=240)
    except Exception as e:
        st.error(f"Failed to open close-up image: {e}")
        closeup_img = None
else:
    closeup_img = None

if blouse_file:
    try:
        blouse_img = Image.open(blouse_file).convert("RGB")
        st.image(blouse_img, caption="Blouse Reference (B)", width=240)
    except Exception as e:
        st.error(f"Failed to open blouse image: {e}")
        blouse_img = None
else:
    blouse_img = None

col1, col2 = st.columns(2)
with col1:
    quality_mode = st.radio("Quality mode", ["A — Ultra Accuracy (slower)", "B — Balanced", "C — Fast"], index=0)
with col2:
    do_upscale = st.checkbox("Apply additional upscale (if available)", value=False)

if st.button("Generate 2K Try-On"):

    if not lehenga_img:
        st.error("Please upload the full-view lehenga image (required).")
    else:
        with st.spinner("Generating strict prompt from provided images..."):
          
            try:
                instruction_prompt = generate_prompt(lehenga_img, closeup_img, blouse_img)
                st.subheader("Generation Instruction Prompt")
                st.write(instruction_prompt)
            except Exception as e:
                st.error(f"Prompt generation failed: {e}")
                instruction_prompt = None

        if instruction_prompt:
            with st.spinner("Generating 2K model image — this may take a while..."):
                image_bytes = generate_image_from_prompt(instruction_prompt)
                if not image_bytes:
                    st.error("Image generation failed. Check API key, model availability, and quota.")
                else:
                    try:
                        out_img = Image.open(BytesIO(image_bytes)).convert("RGB")
                        st.subheader("Final Generated Image (2048×2048)")
                        st.image(out_img, use_column_width=True)

                   
                        buf = BytesIO()
                        out_img.save(buf, format="JPEG", quality=95)
                        buf.seek(0)
                        st.download_button(
                            label="📥 Download Image (JPEG)",
                            data=buf,
                            file_name="model_lehenga_2k.jpg",
                            mime="image/jpeg"
                        )
                    except Exception as e:
                        st.error(f"Failed to display or save generated image: {e}")

        if do_upscale:
            st.info("Upscale requested. If you have an external upscaler (Real-ESRGAN) or a Gemini upscaler model,"
                    "we can add a secondary upscaling pass. Tell me if you want that integrated.")
