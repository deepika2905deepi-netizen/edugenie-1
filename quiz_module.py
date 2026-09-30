"""
Quiz Module - MCQ Quiz Generation using Google Gemini API.
Generates exactly 3 multiple-choice questions with 4 options each.
Returns structured JSON with correct answers.
"""

import json
import re
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


def clean_json_block(text: str) -> str:
    """
    Remove Markdown code fences and extra whitespace from the model response.
    Handles ```json ... ``` and ``` ... ``` formats.

    Args:
        text: Raw text that may contain code fences.

    Returns:
        Cleaned JSON string.
    """
    text = text.strip()

    # Match code fences: ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    return text


def generate_quiz(text: str) -> list:
    """
    Generate exactly 3 MCQs from the given text using Google Gemini.

    Each question contains:
    - question: The question text
    - options: List of 4 options (A, B, C, D)
    - answer: The correct option letter

    Args:
        text: The source text/passage to generate questions from.

    Returns:
        A list of 3 question dictionaries.

    Raises:
        ValueError: If the API key is not configured.
        Exception: If generation or parsing fails.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add your API key to the .env file."
        )

    try:
        # Use the model specified in .env (defaults to gemini-1.5-flash)
        model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = f"""Generate exactly 3 multiple-choice questions based on the following text.
Each question must have exactly 4 options labeled A, B, C, D.
Include the correct answer for each question.

Return ONLY valid JSON in this exact format (no other text):
[
  {{
    "question": "The question text here?",
    "options": ["A. First option", "B. Second option", "C. Third option", "D. Fourth option"],
    "answer": "A"
  }}
]

The "answer" field must be the letter (A, B, C, or D) of the correct option.

Text:
{text}"""

        response = model.generate_content(prompt)

        # Clean markdown code fences and parse JSON safely
        cleaned = clean_json_block(response.text)
        quiz_data = json.loads(cleaned)

        # Validate structure
        if not isinstance(quiz_data, list):
            raise ValueError("Quiz data is not a list")

        if len(quiz_data) != 3:
            raise ValueError(f"Expected 3 questions, got {len(quiz_data)}")

        for i, q in enumerate(quiz_data):
            if "question" not in q or "options" not in q or "answer" not in q:
                raise ValueError(f"Question {i+1} is missing required fields")
            if len(q["options"]) != 4:
                raise ValueError(f"Question {i+1} does not have 4 options")

        return quiz_data

    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse quiz JSON: {str(e)}")
    except Exception as e:
        if "GEMINI_API_KEY" in str(e):
            raise
        raise Exception(f"Failed to generate quiz: {str(e)}")
