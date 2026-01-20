from openai import AzureOpenAI
from app.config import *

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-01"
)

def generate_answer(question, context):
    prompt = f"""
Answer the question using ONLY the context below.
If unsure, say you do not know.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS
    )

    return response.choices[0].message.content