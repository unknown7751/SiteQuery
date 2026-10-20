"""Evaluation harness package. Loads `.env` so GROQ_API_KEY works when run locally."""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional; real environment variables still work
    pass
