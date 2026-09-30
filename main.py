"""
EduGenie - Google Gemini Powered Learning Assistant
Main FastAPI application with REST API endpoints and Jinja2 templating.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import feature modules
from qna import get_answer
from explanation_module import get_explanation
from quiz_module import generate_quiz
from summary_module import get_summary
from learning_path import get_learning_recommendations

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI application
app = FastAPI(
    title="EduGenie",
    description="Google Gemini Powered Learning Assistant",
    version="1.0.0",
)

# Mount static files (CSS, JS) and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Pydantic request models for input validation
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    """Request model for Q&A endpoint."""
    question: str


class TopicRequest(BaseModel):
    """Request model for explanation and learning path endpoints."""
    topic: str


class TextRequest(BaseModel):
    """Request model for quiz and summary endpoints."""
    text: str


# ---------------------------------------------------------------------------
# Frontend Route
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the main EduGenie dashboard page.

    Args:
        request: The incoming HTTP request.

    Returns:
        Rendered HTML template for the main page.
    """
    return templates.TemplateResponse(request, "index.html", {"request": request})


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/qa")
async def qa_endpoint(data: QuestionRequest):
    """
    POST /qa - Answer a student's question using Google Gemini.

    Args:
        data: QuestionRequest containing the question text.

    Returns:
        JSON response with the answer or an error message.
    """
    try:
        if not data.question.strip():
            return JSONResponse(
                {"error": "Question cannot be empty."},
                status_code=400,
            )

        answer = get_answer(data.question.strip())
        return JSONResponse({"answer": answer})

    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/explain")
async def explain_endpoint(data: TopicRequest):
    """
    POST /explain - Get a simple, beginner-friendly explanation.

    Args:
        data: TopicRequest containing the topic to explain.

    Returns:
        JSON response with the explanation or an error message.
    """
    try:
        if not data.topic.strip():
            return JSONResponse(
                {"error": "Topic cannot be empty."},
                status_code=400,
            )

        explanation = get_explanation(data.topic.strip())
        return JSONResponse({"explanation": explanation})

    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/quiz")
async def quiz_endpoint(data: TextRequest):
    """
    POST /quiz - Generate exactly 3 MCQs from the given text.

    Args:
        data: TextRequest containing the source text.

    Returns:
        JSON response with quiz questions or an error message.
    """
    try:
        if not data.text.strip():
            return JSONResponse(
                {"error": "Text cannot be empty."},
                status_code=400,
            )

        quiz = generate_quiz(data.text.strip())
        return JSONResponse({"quiz": quiz})

    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/summarize")
async def summarize_endpoint(data: TextRequest):
    """
    POST /summarize - Summarize a long passage into a concise version.

    Args:
        data: TextRequest containing the text to summarize.

    Returns:
        JSON response with the summary or an error message.
    """
    try:
        if not data.text.strip():
            return JSONResponse(
                {"error": "Text cannot be empty."},
                status_code=400,
            )

        summary = get_summary(data.text.strip())
        return JSONResponse({"summary": summary})

    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/learn/recommendations")
async def learn_recommendations_endpoint(data: TopicRequest):
    """
    POST /learn/recommendations - Get a personalized learning path.

    Args:
        data: TopicRequest containing the topic for recommendations.

    Returns:
        JSON response with the learning path or an error message.
    """
    try:
        if not data.topic.strip():
            return JSONResponse(
                {"error": "Topic cannot be empty."},
                status_code=400,
            )

        recommendations = get_learning_recommendations(data.topic.strip())
        return JSONResponse({"recommendations": recommendations})

    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
