from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

models = client.models.list()
for m in sorted(models.data, key=lambda x: x.id):
    print(m.id)