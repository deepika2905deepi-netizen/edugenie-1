"""
Explanation Module - Simple concept explanations.
Uses a local lightweight model (LaMini-Flan-T5) when available,
with automatic fallback to Google Gemini API.
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


def _try_local_model(topic: str) -> str | None:
    """
    Attempt to generate an explanation using the local LaMini-Flan-T5 model.
    Returns None if the local model is not available.

    Args:
        topic: The concept or topic to explain.

    Returns:
        The generated explanation string, or None if local model unavailable.
    """
    try:
        # Lazy import - only loads when this function is called
        from transformers import pipeline

        # LaMini-Flan-T5-783M: lightweight, instruction-tuned, CPU-compatible
        generator = pipeline(
            "text2text-generation",
            model="LaMini-Flan-T5-783M",
            max_length=512,
        )

        prompt = (
            f"Explain the following topic in simple, beginner-friendly language. "
            f"Avoid complicated terminology and use everyday examples:\n\n{topic}"
        )

        result = generator(prompt, max_length=512, num_return_sequences=1)
        return result[0]["generated_text"].strip()

    except Exception:
        # Local model not available (transformers/torch not installed, etc.)
        # Return None so the caller can fall back to Gemini
        return None


def _fallback_to_gemini(topic: str) -> str:
    """
    Generate an explanation using Google Gemini as a fallback.

    Args:
        topic: The concept or topic to explain.

    Returns:
        The generated explanation string.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add your API key to the .env file."
        )

    # Use the model specified in .env (defaults to gemini-1.5-flash)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""You are EduGenie, a patient and clear teacher.
Explain the following topic in simple, beginner-friendly language.
Avoid unnecessarily complicated terminology.
Use everyday examples and analogies to make it easy to understand.
Keep the explanation concise but thorough.

Topic: {topic}

Explanation:"""

    response = model.generate_content(prompt)
    return response.text.strip()


def get_explanation(topic: str) -> str:
    """
    Get a simple, beginner-friendly explanation of a topic.
    Tries the local model first, falls back to Gemini if unavailable.

    Args:
        topic: The concept or topic to explain.

    Returns:
        A clear, simplified explanation string.

    Raises:
        ValueError: If no AI backend is available.
        Exception: If the generation fails.
    """
    # Try local lightweight model first (privacy-friendly, works offline)
    local_result = _try_local_model(topic)
    if local_result:
        return local_result

    # Fall back to Gemini API
    return _fallback_to_gemini(topic)
