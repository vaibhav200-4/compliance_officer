
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

#_MODEL_NAME = "llama-3.3-70b-versatile"
_MODEL_NAME = "llama-3.1-8b-instant"


def get_llm(temperature: float = 0.2):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Create a .env file in this folder with:\n"
            "GROQ_API_KEY=your_key_here"
        )
    return ChatGroq(model=_MODEL_NAME, temperature=temperature, api_key=api_key)