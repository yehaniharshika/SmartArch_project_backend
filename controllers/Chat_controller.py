"""
SmartArch — controllers/Chat_controller.py

3 endpoints:
  POST /api/chat/<project_id>/ask          — architect asks (JWT required)
  GET  /api/chat/<project_id>/history      — chat history  (JWT required)
  POST /api/chat/share/<token>/ask         — client asks via share link (no JWT)

Postman examples:
  Architect:
    POST http://localhost:5000/api/chat/PRJ-AB1234/ask
    Authorization: Bearer <token>
    Body JSON: {"question": "What is the bedroom size?"}

  Client (share link):
    POST http://localhost:5000/api/chat/share/<share_token>/ask
    Body JSON: {"question": "What is the kitchen area?"}

  History:
    GET http://localhost:5000/api/chat/PRJ-AB1234/history
    Authorization: Bearer <token>
"""
import jwt  # type: ignore  # pylint: disable=import-error
from flask import Blueprint, request, jsonify, g

from config import Config
from dao.FloorPlan_dao import FloorPlanDAO
from services.Chat_service import ChatService
from utils.auth_utils import token_required

chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")


# ── Architect asks (JWT required) ─────────────────────────────
@chat_bp.route("/<string:project_id>/ask", methods=["POST"])
@token_required
def ask_question(project_id: str):
    fp = FloorPlanDAO.get_by_id(project_id)
    if not fp:
        return jsonify({"success": False, "message": "Project not found."}), 404
    if fp.user_id != g.user_id:
        return jsonify({"success": False, "message": "Access denied."}), 403

    data     = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"success": False, "message": "Please provide a question."}), 400

    result, status = ChatService.answer_question(project_id, question)
    return jsonify(result), status


# ── Chat history (JWT required) ────────────────────────────────
@chat_bp.route("/<string:project_id>/history", methods=["GET"])
@token_required
def get_chat_history(project_id: str):
    fp = FloorPlanDAO.get_by_id(project_id)
    if not fp:
        return jsonify({"success": False, "message": "Project not found."}), 404
    if fp.user_id != g.user_id:
        return jsonify({"success": False, "message": "Access denied."}), 403

    result, status = ChatService.get_history(project_id)
    return jsonify(result), status


# ── Client asks via share link (NO JWT needed) ─────────────────
@chat_bp.route("/share/<string:token>/ask", methods=["POST"])
def client_ask(token: str):
    """
    This is the MAIN CLIENT-FACING endpoint.
    Client opens the share link → frontend calls this endpoint.
    No login required — the share token identifies the project.
    """
    try:
        payload    = jwt.decode(token, Config.JWT_SECRET,
                                algorithms=[Config.JWT_ALGORITHM])
        project_id = payload.get("project_id")
    except jwt.ExpiredSignatureError:
        return jsonify({"success": False,
                        "message": "This share link has expired."}), 401
    except jwt.InvalidTokenError:
        return jsonify({"success": False,
                        "message": "Invalid share link."}), 401

    if not project_id:
        return jsonify({"success": False, "message": "Invalid share token."}), 401

    fp = FloorPlanDAO.get_by_id(project_id)
    if not fp:
        return jsonify({"success": False, "message": "Floor plan not found."}), 404

    if fp.status != "ready":
        return jsonify({
            "success": False,
            "message": "This floor plan is still being processed. Please try again shortly.",
        }), 202

    data     = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"success": False, "message": "Please provide a question."}), 400

    result, status = ChatService.answer_question(project_id, question)
    return jsonify(result), status


# ── Clear history (optional, for testing) ─────────────────────
@chat_bp.route("/<string:project_id>/history", methods=["DELETE"])
@token_required
def clear_history(project_id: str):
    fp = FloorPlanDAO.get_by_id(project_id)
    if not fp:
        return jsonify({"success": False, "message": "Project not found."}), 404
    if fp.user_id != g.user_id:
        return jsonify({"success": False, "message": "Access denied."}), 403

    from dao.FloorPlan_dao import FloorPlanDAO as FPD
    from entity.ChatMessage_entity import ChatMessage
    from db.database import db
    ChatMessage.query.filter_by(project_id=project_id).delete()
    db.session.commit()
    return jsonify({"success": True, "message": "Chat history cleared."}), 200