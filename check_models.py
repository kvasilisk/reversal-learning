from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

models = client.models.list()
for m in sorted(models.data, key=lambda x: x.id):
    print(m.id)