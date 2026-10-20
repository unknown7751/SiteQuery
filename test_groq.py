"""Standalone smoke test: LangChain ChatGroq + openai/gpt-oss-120b on Groq."""

import os
import sys

from dotenv import load_dotenv
from langchain_groq import ChatGroq

MODEL = "openai/gpt-oss-120b"
PROMPT = (
    "Explain the purpose of a Request for Information (RFI) in construction "
    "in 2 sentences."
)


def main() -> int:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set (expected in .env).", file=sys.stderr)
        return 1

    llm = ChatGroq(model=MODEL, temperature=0)
    response = llm.invoke(PROMPT)
    print(f"Model: {MODEL}\n")
    print(response.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
