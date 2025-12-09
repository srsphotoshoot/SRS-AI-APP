import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    print("Google API key not found in .env")
    exit()

genai.configure(api_key=GEMINI_API_KEY)

# List all models
models = genai.list_models()
print(f"Found {len(models)} models:\n")

for m in models:
    print("Name:", m.name)
    print("Display Name:", getattr(m, "display_name", "N/A"))
    print("Type:", getattr(m, "model_type", "N/A"))
    print("Supports Image Generation:", "image" in m.name.lower() or "flash" in m.name.lower())
    print("-" * 50)
