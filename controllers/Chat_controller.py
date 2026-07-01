"""
SmartArch — controllers/Chat_controller.py
PRESENTATION LAYER — HTTP only. No business logic here.

These endpoints are used by the CLIENT (not the architect), so they
do NOT require @token_required login. Instead, security comes from
needing a valid project_id from a working share link.

2 Endpoints:
  1. POST /api/chat/<project_id>          ← client asks a question
  2. GET  /api/chat/<project_id>/history  ← (optional) view past Q&A
"""
from flask import Blueprint, request, jsonify

from services.Chat_service import ChatService

chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")


# ══════════════════════════════════════════════════════════
# CONTROLLER 1 — Ask a question about a floor plan
# ══════════════════════════════════════════════════════════
@chat_bp.route("/<string:project_id>", methods=["POST"])
def ask_question(project_id: str):
    """
    POST /api/chat/<project_id>
    Body (raw JSON):
      { "question": "What is the width of the kitchen?" }

    ── Postman ──────────────────────────────────────────────
    Method  : POST
    URL     : http://localhost:5000/api/chat/PRJ-75F29F
    Headers : Content-Type: application/json
    Body    : raw → JSON
      { "question": "What is the width of the kitchen?" }
    ─────────────────────────────────────────────────────────
    No Authorization header needed — this is the public client-facing
    endpoint, reached via the share link.
    """
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")

    result, status_code = ChatService.answer_question(project_id, question)
    return jsonify(result), status_code


# ══════════════════════════════════════════════════════════
# CONTROLLER 2 — Get past chat history for a floor plan
# ══════════════════════════════════════════════════════════
@chat_bp.route("/<string:project_id>/history", methods=["GET"])
def get_history(project_id: str):
    """
    GET /api/chat/<project_id>/history

    ── Postman ──────────────────────────────────────────────
    Method : GET
    URL    : http://localhost:5000/api/chat/PRJ-75F29F/history
    ─────────────────────────────────────────────────────────
    """
    result, status_code = ChatService.get_history(project_id)
    return jsonify(result), status_code