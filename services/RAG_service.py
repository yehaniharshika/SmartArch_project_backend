"""
SmartArch — services/RAG_service.py

Stores extracted floor plan data into ChromaDB and retrieves
relevant chunks when a client asks a question.

HOW IT WORKS:
  Store:    room data → text sentences → Gemini embeddings → ChromaDB
  Retrieve: question  → Gemini embedding → similarity search → context chunks
"""
import os
import chromadb
from chromadb.config import Settings


# ── ChromaDB persistent client ─────────────────────────────────
_chroma_client = None


def _get_client():
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    persist_dir = os.getenv("VECTORSTORE_DIR", "vectorstore")
    os.makedirs(persist_dir, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    print(f"[RAG] ChromaDB client ready — {persist_dir}")
    return _chroma_client


def _get_collection(project_id: str):
    """Each project gets its own ChromaDB collection."""
    client = _get_client()
    name = f"plan_{project_id.replace('-', '_').lower()}"
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ── Gemini Embedding ───────────────────────────────────────────
def _embed(texts: list, task_type: str = "RETRIEVAL_DOCUMENT") -> list:
    """Converts text list to embedding vectors using Gemini."""
    from google import genai
    from google.genai import types

    api_key    = os.getenv("GEMINI_API_KEY")
    embed_model= os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
    client     = genai.Client(api_key=api_key)

    embeddings = []
    for text in texts:
        result = client.models.embed_content(
            model=embed_model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        embeddings.append(result.embeddings[0].values)
    return embeddings


# ── STORE — called right after analysis finishes ───────────────
def store_floor_plan_data(project_id: str, project_name: str,
                          rooms: list, total_area_sqft: float,
                          detections: list) -> int:
    """
    Converts room/detection data into text sentences and stores
    them in ChromaDB with Gemini embeddings.
    Returns number of documents stored.
    """
    collection = _get_collection(project_id)

    # Clear old data for this project (handles re-uploads)
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    documents = []
    ids       = []
    metadatas = []

    # ── Per-room documents ─────────────────────────────────────
    for i, room in enumerate(rooms):
        name    = room.get("name", f"Room {i+1}")
        w_ft    = room.get("width_ft_in",  "unknown")
        h_ft    = room.get("height_ft_in", "unknown")
        w_m     = room.get("width_m",  0)
        h_m     = room.get("height_m", 0)
        sqft    = room.get("area_sqft", 0)
        sqm     = room.get("area_sqm",  0)
        rtype   = room.get("room_type", "room")
        source  = room.get("dimension_source", "")
        notes   = room.get("notes", "")

        sentence = (
            f"{name} is a {rtype}. "
            f"Its width is {w_ft} ({w_m} meters) and "
            f"its length is {h_ft} ({h_m} meters). "
            f"The area of {name} is {sqft} square feet ({sqm} square meters)."
        )

        if source == "ocr_exact_match":
            sentence += " Dimensions were read directly from the plan."
        elif source == "wall_geometry_estimate":
            sentence += " Dimensions were estimated from wall geometry."
        elif source == "unmatched":
            sentence += " Exact dimensions could not be determined."

        if notes:
            sentence += f" Note: {notes}"

        documents.append(sentence)
        ids.append(f"{project_id}_room_{i}")
        metadatas.append({
            "project_id": project_id,
            "room_name":  name,
            "type":       "room",
        })

    # ── Total area document ────────────────────────────────────
    if total_area_sqft and total_area_sqft > 0:
        total_sqm = round(total_area_sqft * 0.092903, 2)
        documents.append(
            f"The total floor area of the {project_name} floor plan is "
            f"{total_area_sqft} square feet ({total_sqm} square meters)."
        )
        ids.append(f"{project_id}_total_area")
        metadatas.append({"project_id": project_id, "type": "total_area"})

    # ── Structural counts document ─────────────────────────────
    if detections:
        # detections can be DetectionDTO objects or dicts
        def get_label(d):
            return d.label if hasattr(d, "label") else d.get("label", "")

        doors   = sum(1 for d in detections if get_label(d) == "door")
        windows = sum(1 for d in detections if get_label(d) == "window")
        walls   = sum(1 for d in detections if get_label(d) == "wall")

        documents.append(
            f"This floor plan has {len(rooms)} rooms, {doors} doors, "
            f"{windows} windows, and {walls} wall segments detected."
        )
        ids.append(f"{project_id}_structural")
        metadatas.append({"project_id": project_id, "type": "structural"})

    # ── Design suggestions document ────────────────────────────
    suggestions = _build_design_suggestions(rooms, project_name)
    if suggestions:
        documents.append(suggestions)
        ids.append(f"{project_id}_design")
        metadatas.append({"project_id": project_id, "type": "design"})

    if not documents:
        print(f"[RAG] No documents to store for {project_id}")
        return 0

    print(f"[RAG] Embedding {len(documents)} documents for {project_id} ...")
    embeddings = _embed(documents, task_type="RETRIEVAL_DOCUMENT")

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"[RAG] ✅ Stored {len(documents)} documents for {project_id}")
    return len(documents)


def _build_design_suggestions(rooms: list, project_name: str) -> str:
    """Builds a design suggestions text chunk based on extracted rooms."""
    if not rooms:
        return ""

    suggestions = [f"Design suggestions for the {project_name} floor plan:"]

    bedrooms = [r for r in rooms if r.get("room_type") == "bedroom"]
    bathrooms= [r for r in rooms if r.get("room_type") == "bathroom"]
    kitchens = [r for r in rooms if r.get("room_type") == "kitchen"]
    living   = [r for r in rooms if r.get("room_type") == "living"]

    if bedrooms:
        suggestions.append(
            f"There are {len(bedrooms)} bedroom(s). "
            "Bedrooms should be positioned away from high-traffic areas "
            "for better privacy and noise reduction."
        )

    if bathrooms:
        suggestions.append(
            f"The plan has {len(bathrooms)} bathroom(s). "
            "Bathrooms should ideally be adjacent to bedrooms and near "
            "plumbing access points for cost efficiency."
        )

    if kitchens:
        suggestions.append(
            "The kitchen should have adequate ventilation and natural light. "
            "Position it near the dining area to improve workflow efficiency."
        )

    if living:
        suggestions.append(
            "The living area should receive maximum natural light. "
            "Consider large windows or an open-plan layout for better airflow."
        )

    small_rooms = [
        r for r in rooms
        if 0 < r.get("area_sqft", 0) < 80
    ]
    if small_rooms:
        names = ", ".join(r.get("name", "room") for r in small_rooms)
        suggestions.append(
            f"The following rooms are relatively small (under 80 sq ft): {names}. "
            "Consider built-in storage and space-saving furniture designs."
        )

    return " ".join(suggestions)


# ── SEARCH — called when client asks a question ────────────────
def search_floor_plan_data(project_id: str, question: str,
                           top_k: int = 3) -> str:
    """
    Embeds the question, searches ChromaDB for the most relevant
    floor plan data, and returns it as a single context string
    for the LLM to use when generating an answer.
    """
    try:
        collection = _get_collection(project_id)
        count = collection.count()
        if count == 0:
            print(f"[RAG] No data in collection for {project_id}")
            return ""

        query_embedding = _embed([question], task_type="RETRIEVAL_QUERY")[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
        )

        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""

        print(f"[RAG] Retrieved {len(docs)} chunks for: '{question}'")
        return "\n".join(docs)

    except Exception as e:
        print(f"[RAG] Search failed for {project_id}: {e}")
        return ""


# ── DELETE — called when project is deleted ────────────────────
def delete_floor_plan_data(project_id: str) -> None:
    """Removes the ChromaDB collection for a deleted project."""
    try:
        client = _get_client()
        name = f"plan_{project_id.replace('-', '_').lower()}"
        client.delete_collection(name=name)
        print(f"[RAG] Deleted collection for {project_id}")
    except Exception as e:
        print(f"[RAG] Could not delete collection: {e}")