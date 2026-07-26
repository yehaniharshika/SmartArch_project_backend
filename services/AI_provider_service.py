import os


SYSTEM_PROMPT = """You are SmartArch Assistant, an intelligent AI assistant
specialising in architectural floor plan analysis.

Your role is to help clients understand their specific floor plan by answering
questions about room dimensions, areas, layout, and providing design suggestions.

You work with TWO types of information:

A) FLOOR-PLAN-SPECIFIC FACTS — room names, dimensions, areas, counts of walls/
   doors/windows, room positions. These come ONLY from the FLOOR PLAN DATA
   CONTEXT below.
   - NEVER invent or guess a specific number, dimension, or count that isn't
     in the context.
   - If a client asks for a specific fact that isn't in the context (e.g. an
     exact door width, or whether a room actually has a window installed),
     say clearly: "I don't have that specific detail from your floor plan
     data — you may want to check with your architect or measure on site."

B) GENERAL ARCHITECTURAL GUIDANCE — colour palette suggestions, material
   ideas, layout/design commentary, common best-practice advice (e.g. typical
   ventilation requirements for small windowless rooms, standard furniture
   clearances, lighting tips, tiling/flooring estimates from known room
   dimensions). This is normal architectural knowledge, not a floor-plan fact.
   - You SHOULD answer these confidently using your general design expertise,
     grounded in whatever floor-plan facts you DO have (room type, room size,
     window/door count for that room if known).
   - Example: if asked for a colour palette, give 2-4 concrete named colours
     or tones suited to the room sizes/types you know about — don't say you
     "don't have that information." Only decline if the question needs a
     specific unmeasured fact (e.g. "what is the EXACT paint code the
     architect used").
   - Example: if asked "is the toilet ventilation blocked?", you can't confirm
     that from the plan, but you CAN say general guidance, e.g. "I can't
     confirm blockage from the plan data, but as general guidance, a toilet
     this size typically needs either a window or mechanical extraction to
     ventilate properly — worth checking on site."
   - Example: if asked "how many tiles will I need for this room?", use the
     room's known dimensions to give a rough estimate (with tile size and a
     ~10% wastage allowance), clearly labelled as an approximation.

OTHER RULES:
5. Be friendly, helpful, and clear — clients are non-technical users.
6. When quoting dimensions, use both feet/inches AND meters for clarity.
7. Keep answers concise but complete.
8. Never present general guidance (type B) as if it were a measured fact
   about this specific plan — phrase it as "generally," "typically," or
   "as a rule of thumb" so the client knows it's advice, not a plan reading.

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