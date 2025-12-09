
import streamlit as st
from PIL import Image
import google.generativeai as genai
import base64
from dotenv import load_dotenv
import os
import io

load_dotenv()

GEMINI_API_KEY= os.getenv("GOOGLE_API_KEY")


genai.configure(api_key=GEMINI_API_KEY)

MODEL = "gemini-2.0-flash-exp"  


# ---------------------------
# Utility: Convert PIL → base64 → inlineData
# ---------------------------
def image_to_inline_data(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return {
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(buffered.getvalue()).decode()
        }
    }


# ---------------------------
# Step 1 — Generate Instruction Prompt
# ---------------------------
def generate_instruction_prompt(lehenga_img, closeup_img):
    prompt = """
Analyze the two provided images:

1. Full lehenga image
2. Close-up embroidery / design image

Your task:
- Describe the lehenga in extreme detail (fabric, embroidery, blouse, dupatta, borders, textures).
- Combine details from BOTH images.
- Ensure maximum accuracy for colours, patterns and stone-work.

Then output ONLY the following final instruction for image generation:

"Generate a 2K ultra realistic image of a model wearing this EXACT SAME LEHENGA. 
Match embroidery, blouse design, dupatta style, colours, fabric shine, skirt flare, 
and all fine textures EXACTLY. No creative changes. Hyper-real, studio photography."

OUTPUT FORMAT:
Return ONLY the final instruction. No description, no explanation.
"""

    model = genai.GenerativeModel(MODEL)

    response = model.generate_content(
        [
            {"text": prompt},
            image_to_inline_data(lehenga_img),
            image_to_inline_data(closeup_img)
        ],
        stream=False
    )

    return response.text


# ---------------------------
# Step 2 — Generate final TRY-ON image
# ---------------------------
def generate_final_image(instruction):
    model = genai.GenerativeModel(MODEL)

    response = model.generate_content(
        [{"text": instruction}],
        generation_config={
            "response_mime_type": "image/jpeg",
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 8192
        },
        stream=False
    )

    # Extract base64 image
    img_base64 = response.candidates[0].content.parts[0].inline_data.data
    return base64.b64decode(img_base64)


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("👗 Ultra Accurate Lehenga Try-On Generator (Gemini 2.0 Flash EXP)")

lehenga_file = st.file_uploader("Upload Lehenga Full Image", type=["jpg", "jpeg", "png"])
closeup_file = st.file_uploader("Upload Close-up / Embroidery Image", type=["jpg", "jpeg", "png"])

if lehenga_file and closeup_file:
    lehenga_img = Image.open(lehenga_file)
    closeup_img = Image.open(closeup_file)

    st.image(lehenga_img, caption="Lehenga Image", width=300)
    st.image(closeup_img, caption="Close-up Image", width=300)

    if st.button("Generate Try-On Image"):
        with st.spinner("Generating hyper-accurate instruction prompt..."):
            instruction = generate_instruction_prompt(lehenga_img, closeup_img)

        st.subheader("Generated Instruction Prompt")
        st.code(instruction)

        with st.spinner("Generating 2K Try-On Image..."):
            output_bytes = generate_final_image(instruction)

        st.subheader("Final Generated Try-On")
        st.image(output_bytes, use_column_width=True)

        st.download_button(
            label="📥 Download 2K Image",
            data=output_bytes,
            file_name="lehenga_model_2k.jpg",
            mime="image/jpeg"
        )
