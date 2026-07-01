"""
SmartArch — services/rag_service.py

This file's job: take the room data we extracted (Step 9 of the
pipeline) and make it SEARCHABLE by meaning, using ChromaDB.

Think of ChromaDB like a smart filing cabinet:
  - We write each room's info on a "card" (a sentence)
  - We file each card under a number-code that represents its MEANING
  - When a client asks a question, we convert the question into the
    same kind of number-code, and pull out the closest-matching cards
"""
import chromadb
from config import Config
from services.AI_provider_service import embed_text

# One ChromaDB client for the whole app, stored on disk at
# the path set in Config.CHROMA_PATH (backend/vectorstore/)
_chroma_client = chromadb.PersistentClient(path=Config.CHROMA_PATH)


def _get_collection(project_id: str):
    """
    Each floor plan gets its OWN ChromaDB "collection" (like a separate
    folder), named after its project_id. This keeps Kitchen data from
    "PRJ-AAA111" from ever mixing with Kitchen data from "PRJ-BBB222".
    """
    collection_name = f"plan_{project_id.replace('-', '_').lower()}"
    return _chroma_client.get_or_create_collection(name=collection_name)


# ══════════════════════════════════════════════════════════
# STORE — called right after analysis finishes (Step 9 → Step 11)
# ══════════════════════════════════════════════════════════
def store_floor_plan_data(project_id: str, project_name: str,
                          rooms: list, total_area_sqft: float,
                          detections: list) -> int:
    """
    Converts room data into searchable "documents" and stores them
    in ChromaDB. Returns the number of documents stored.

    rooms = the list Gemini gave us, e.g.
      [{"name": "Kitchen", "width_ft_in": "11' 6\"",
        "length_ft_in": "9' 2\"", "area_sqft": 105.4, ...}, ...]
    """
    collection = _get_collection(project_id)

    documents  = []   # the actual sentences
    ids        = []   # unique ID for each sentence
    metadatas  = []    # extra tags (which room, which project)

    # ── One document per room ───────────────────────────────
    for i, room in enumerate(rooms):
        name   = room.get("name", "Unknown Room")
        w_ft   = room.get("width_ft_in", "unknown")
        l_ft   = room.get("length_ft_in", "unknown")
        w_m    = room.get("width_m", "unknown")
        l_m    = room.get("length_m", "unknown")
        sqft   = room.get("area_sqft", 0)
        sqm    = room.get("area_sqm", 0)
        floor  = room.get("floor", "Ground")
        notes  = room.get("notes", "")

        sentence = (
            f"{name} is located on the {floor} floor. "
            f"The width of {name} is {w_ft} ({w_m} meters) and "
            f"the length of {name} is {l_ft} ({l_m} meters). "
            f"The total area of {name} is {sqft} square feet "
            f"({sqm} square meters)."
        )
        if notes:
            sentence += f" Note: {notes}."

        documents.append(sentence)
        ids.append(f"{project_id}_room_{i}")
        metadatas.append({
            "project_id": project_id,
            "room_name": name,
            "type": "room",
        })

    # ── One document for the total area ─────────────────────
    if total_area_sqft:
        documents.append(
            f"The total floor area of the entire {project_name} plan "
            f"is {total_area_sqft} square feet."
        )
        ids.append(f"{project_id}_total_area")
        metadatas.append({"project_id": project_id, "type": "total_area"})

    # ── One document per door (for "is the door position good?" questions) ──
    doors = [d for d in detections if d.label == "door"]
    for i, door in enumerate(doors):
        sentence = (
            f"There is a door with width {door.width_m} meters "
            f"and height {door.height_m} meters."
        )
        documents.append(sentence)
        ids.append(f"{project_id}_door_{i}")
        metadatas.append({"project_id": project_id, "type": "door"})

    # ── One document for door/window counts ─────────────────
    documents.append(
        f"This floor plan has {len(doors)} doors and "
        f"{len([d for d in detections if d.label == 'window'])} windows in total."
    )
    ids.append(f"{project_id}_counts")
    metadatas.append({"project_id": project_id, "type": "counts"})

    if not documents:
        print(f"[RAG] No room data to store for {project_id}")
        return 0

    # ── Convert every sentence into numbers (embeddings) ────
    print(f"[RAG] Embedding {len(documents)} documents for {project_id} ...")
    embeddings = [embed_text(doc, task_type="retrieval_document") for doc in documents]

    # ── Save into ChromaDB ───────────────────────────────────
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"[RAG] ✅ Stored {len(documents)} documents in ChromaDB for {project_id}")
    return len(documents)


# ══════════════════════════════════════════════════════════
# SEARCH — called when a client asks a question
# ══════════════════════════════════════════════════════════
def search_floor_plan_data(project_id: str, question: str, top_k: int = 3) -> str:
    """
    Converts the client's question into numbers, finds the closest-
    matching stored sentences, and returns them joined as one block
    of text (this becomes the "context" given to Gemini for the
    final answer).
    """
    try:
        collection = _get_collection(project_id)
    except Exception as e:
        print(f"[RAG] Collection not found for {project_id}: {e}")
        return ""

    question_embedding = embed_text(question, task_type="retrieval_query")

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )

    matched_documents = results.get("documents", [[]])[0]
    if not matched_documents:
        return ""

    print(f"[RAG] Found {len(matched_documents)} matching documents for question: '{question}'")
    return "\n".join(matched_documents)


def delete_floor_plan_data(project_id: str) -> None:
    """Called when a project is deleted — cleans up its ChromaDB collection too."""
    try:
        collection_name = f"plan_{project_id.replace('-', '_').lower()}"
        _chroma_client.delete_collection(name=collection_name)
        print(f"[RAG] Deleted ChromaDB collection for {project_id}")
    except Exception as e:
        print(f"[RAG] Could not delete collection (may not exist): {e}")