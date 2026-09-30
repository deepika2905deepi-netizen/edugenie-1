"""
QnA Module - Question Answering using Google Gemini API.
Handles general knowledge and academic questions with precision.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API with the key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    genai.configure(api_key=GEMINI_API_KEY)


def get_answer(question: str) -> str:
    """
    Get an answer to a user's question using Google Gemini.

    Args:
        question: The user's question text.

    Returns:
        A concise, accurate answer string.

    Raises:
        ValueError: If the API key is not configured.
        Exception: If the API call fails.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add your API key to the .env file."
        )

    try:
        # Use the model specified in .env (defaults to gemini-1.5-flash)
        model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = f"""You are EduGenie, a helpful educational assistant for students.
Answer the following question clearly and concisely.
Provide accurate information in a student-friendly manner.

Question: {question}

Answer:"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        raise Exception(f"Failed to generate answer: {str(e)}")
