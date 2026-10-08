"""List the Gemini models your API key can use for text generation.

Run from the project folder:  python scripts/list_gemini_models.py
Copy one of the names (without "models/") into GEMINI_MODEL in .env.
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GOOGLE_API_KEY is not set. Add it to the .env file first.")

client = genai.Client(api_key=api_key)
for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(model.name.removeprefix("models/"))
