from pathlib import Path
import os
from dotenv import load_dotenv
import google.generativeai as genai

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=False)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=api_key)

for m in genai.list_models():
    print(m.name)