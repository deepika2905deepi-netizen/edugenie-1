"""
Learning Path Module - Personalized Learning Recommendations.
Uses Google Gemini to generate structured, beginner-to-advanced learning paths.
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


def get_learning_recommendations(topic: str) -> str:
    """
    Generate a personalized, structured learning path for a given topic.
    Includes beginner-to-advanced steps, difficulty levels, and resources.

    Args:
        topic: The subject or topic to create a learning path for.

    Returns:
        A formatted learning path with steps, topics, and resources.

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

        prompt = f"""You are EduGenie, an expert educational advisor.
Create a personalized, structured learning path for the topic: "{topic}"

The learning path must include:
1. Step-by-step guidance from beginner to advanced level
2. Specific topics to learn at each step
3. Difficulty level for each step (Beginner, Intermediate, Advanced)
4. Suggested resources (videos, articles, books, courses)
5. Estimated time for each step

Format the output clearly with numbered steps, headers, and bullet points.
Make it actionable and motivating for the learner.

Topic: {topic}

Learning Path:"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        raise Exception(f"Failed to generate learning path: {str(e)}")
