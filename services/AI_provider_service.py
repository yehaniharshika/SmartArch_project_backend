"""
SmartArch — services/AI_provider_service.py

Handles all calls to the AI (Gemini) for:
  1. generate_chat_answer() — generates an answer to a client question
  2. embed_text()           — converts text to an embedding vector
                              (used by RAG_service.py)
"""
import os


SYSTEM_PROMPT = """You are SmartArch Assistant, an intelligent AI assistant
specialising in architectural floor plan analysis.

Your role is to help clients understand their specific floor plan by answering
questions about room dimensions, areas, layout, and providing design suggestions.

STRICT RULES:
1. Answer ONLY based on the floor plan data provided in the context below.
2. Do NOT invent or guess dimensions, areas, or room details not in the context.
3. If the information is not available, say clearly:
   "I don't have that specific information from your floor plan."
4. You CAN provide general architectural design suggestions based on the rooms
   and dimensions you DO know about.
5. Be friendly, helpful, and clear - clients are non-technical users.
6. When quoting dimensions, use both feet/inches AND meters for clarity.
7. Keep answers concise but complete.

FLOOR PLAN DATA CONTEXT:
{context}
"""


def generate_chat_answer(question: str, context_text: str,
                         project_name: str = "this floor plan") -> str:
    """
    Calls Gemini LLM with the retrieved context and client question.
    Returns the generated answer as a string.
    """
    from google import genai
    from google.genai import types

    api_key    = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    system = SYSTEM_PROMPT.format(context=context_text or "No floor plan data available.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=[question],
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.3,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list:
    """
    Converts a single text string into a Gemini embedding vector.
    Used by RAG_service.py for both storing and searching.

    task_type options:
      "RETRIEVAL_DOCUMENT" — for storing floor plan data
      "RETRIEVAL_QUERY"    — for embedding client questions

    NOTE: "text-embedding-004" / "embedding-001" were retired by Google.
    The current supported embedding model is "gemini-embedding-001".
    """
    from google import genai
    from google.genai import types

    api_key     = os.getenv("GEMINI_API_KEY")
    embed_model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=embed_model,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return result.embeddings[0].values