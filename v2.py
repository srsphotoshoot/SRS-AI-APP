import streamlit as st
from PIL import Image
import google.generativeai as genai
import base64
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY= os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

VISION_MODEL = "models/gemini-1.0-basic"          
IMAGE_MODEL  = "models/gemini-2.5-flash-image"        
    


def generate_prompt(img):
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
    model = genai.GenerativeModel(VISION_MODEL)
    result = model.generate_content([PROMPT_TEMPLATE, img])
    return result.text

def generate_image(prompt):
    model = genai.GenerativeModel(IMAGE_MODEL)
    result = model.generate_content(prompt, stream=False)
    image_base64 = result.candidates[0].content.parts[0].inline_data.data
    return base64.b64decode(image_base64)


st.title("👗 Lehenga Try-On Generator (Gemini)")

uploaded_file = st.file_uploader("Upload Lehenga Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    input_img = Image.open(uploaded_file)
    st.image(input_img, caption="Uploaded Lehenga", width=350)

    if st.button("Generate Model Image"):
        with st.spinner("Generating prompt from image..."):
            prompt = generate_prompt(input_img)
        st.subheader("Generated Prompt")
        st.write(prompt)

        with st.spinner("Generating model wearing lehenga..."):
            output_image_bytes = generate_image(prompt)

        st.subheader("Final Generated Image")
        st.image(output_image_bytes, use_column_width=True)

        # Download button
        st.download_button(
            label="📥 Download Image",
            data=output_image_bytes,
            file_name="model_lehenga.jpg",
            mime="image/jpeg"
        )
