"""
Summary Module - Text Summarization using Google Gemini API.
Condenses long educational passages into concise, easy-to-understand summaries.
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


def get_summary(text: str) -> str:
    """
    Summarize a long passage into a concise version using Google Gemini.
    Retains core information while eliminating redundancy.

    Args:
        text: The long text/passage to summarize.

    Returns:
        A concise summary string.

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

        prompt = f"""You are EduGenie, an educational assistant that helps students study efficiently.
Summarize the following text into a concise, easy-to-understand version.
Retain all key points and core information while eliminating redundancy.
Use clear, student-friendly language.

Text to summarize:
{text}

Summary:"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        raise Exception(f"Failed to generate summary: {str(e)}")
