"""
SmartArch — FloorPlan_controller.py
CONTROLLER LAYER (Presentation) — HTTP only. Zero business logic.

5 Controllers:
  1. POST   /api/floor-plan/upload          ← upload + analyze
  2. GET    /api/floor-plan/<project_id>    ← get full result
  3. GET    /api/floor-plan/my-plans        ← list logged-in user's plans
  4. DELETE /api/floor-plan/<project_id>    ← delete plan
  5. GET    /api/floor-plan/share/<token>   ← client share view

Postman test:
  1. POST  http://localhost:5000/api/floor-plan/upload
     Headers: Authorization: Bearer <token>   (from /api/auth/login)
     Body → form-data
       project_name  :  Villa Serenova    (text)
       file          :  [your image/pdf]  (file)
     (user_id is NOT sent — it comes from the JWT token)
"""

import jwt  # type: ignore  # pylint: disable=import-error
from flask import Blueprint, request, jsonify, g
from pathlib import Path

from config import Config
from dao.FloorPlan_dao import FloorPlanDAO
from utils.auth_utils import token_required

# Import the service layer with a clear error message if any
# extraction service file is missing from services/extraction/
try:
    from services.FloorPlan_service import FloorPlanService
except ImportError as e:
    raise ImportError(
        "Could not import FloorPlanService. This usually means one of "
        "the extraction service files is missing from "
        "backend/services/extraction/ — check that yolo_service.py, "
        "ocr_service.py, scale_service.py, room_boundary_service.py, "
        "room_parser_service.py, and area_service.py all exist there, "
        f"and that services/extraction/__init__.py exists. "
        f"Original error: {e}"
    ) from e

floor_plan_bp = Blueprint("floor_plan", __name__, url_prefix="/api/floor-plan")


# ══════════════════════════════════════════════════════════
# CONTROLLER 1 — Upload & Analyze
# ══════════════════════════════════════════════════════════
@floor_plan_bp.route("/upload", methods=["POST"])
@token_required
def upload_and_analyze():
    """
    POST /api/floor-plan/upload
    Headers:
      Authorization : Bearer <jwt_token_from_login>
    multipart/form-data:
      project_name  (text, required)
      file          (file, required: PNG/JPG/JPEG/PDF)

    ── Postman ──────────────────────────────────────────────
    Method  : POST
    URL     : http://localhost:5000/api/floor-plan/upload
    Headers : Authorization: Bearer <token>   ← from /api/auth/login response
    Body    : form-data
      Key: project_name   Type: Text   Value: My House Plan
      Key: file            Type: File   Value: [select file]
    ─────────────────────────────────────────────────────────
    user_id is NOT sent in form data — it's decoded from the JWT
    by @token_required and injected as g.user_id.
    """
    user_id = g.user_id   # injected by @token_required

    project_name = request.form.get("project_name", "").strip()
    if not project_name:
        return jsonify({"success": False,
                        "message": "project_name is required."}), 400

    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"success": False,
                        "message": "A floor plan file (PNG/JPG/JPEG/PDF) is required."}), 400

    file = request.files["file"]

    result, status_code = FloorPlanService.upload_and_analyze(
        user_id, project_name, file
    )
    return jsonify(result), status_code


# CONTROLLER 2 — Get Full Result by project_id
@floor_plan_bp.route("/<string:project_id>", methods=["GET"])
@token_required
def get_floor_plan(project_id: str):
    """
    GET /api/floor-plan/<project_id>
    Headers: Authorization: Bearer <token>
    Returns full analysis result for a project.

    Postman 
    Method  : GET
    URL     : http://localhost:5000/api/floor-plan/PRJ-AB1C2D
    Headers : Authorization: Bearer <token>
    """
    fp = FloorPlanDAO.get_by_id(project_id)
    if not fp:
        return jsonify({"success": False,
                        "message": f"Project '{project_id}' not found."}), 404

    if fp.user_id != g.user_id:
        return jsonify({"success": False,
                        "message": "You do not have access to this project."}), 403

    detections = FloorPlanDAO.get_detections(project_id)
    ocr = FloorPlanDAO.get_ocr(project_id)

    return jsonify({
        "success": True,
        "data": {
            **fp.to_dict(),
            "detections": [d.to_dict() for d in detections],
            "ocr": ocr.to_dict() if ocr else None,
        }
    }), 200


# CONTROLLER 3 — List All Plans for the Logged-In User
@floor_plan_bp.route("/my-plans", methods=["GET"])
@token_required
def get_user_plans():
    """
    GET /api/floor-plan/my-plans
    Headers: Authorization: Bearer <token>
    Returns all floor plans uploaded by the LOGGED-IN user (from token),
    newest first.

    Postman
    Method  : GET
    URL     : http://localhost:5000/api/floor-plan/my-plans
    Headers : Authorization: Bearer <token>
    """
    plans = FloorPlanDAO.get_by_user(g.user_id)
    return jsonify({
        "success": True,
        "count": len(plans),
        "data": [fp.to_dict() for fp in plans],
    }), 200


# ══════════════════════════════════════════════════════════
# CONTROLLER 4 — Delete a Floor Plan
@floor_plan_bp.route("/<string:project_id>", methods=["DELETE"])
@token_required
def delete_floor_plan(project_id: str):
    """
    DELETE /api/floor-plan/<project_id>
    Headers: Authorization: Bearer <token>
    Removes the plan and all related detections/OCR/chat from DB.
    Also deletes uploaded files from disk.

    ── Postman ──────────────────────────────────────────────
    Method  : DELETE
    URL     : http://localhost:5000/api/floor-plan/PRJ-AB1C2D
    Headers : Authorization: Bearer <token>
    ─────────────────────────────────────────────────────────
    """
    fp = FloorPlanDAO.get_by_id(project_id)
    if not fp:
        return jsonify({"success": False,
                        "message": f"Project '{project_id}' not found."}), 404

    if fp.user_id != g.user_id:
        return jsonify({"success": False,
                        "message": "You do not have access to this project."}), 403

    for path_attr in ["file_path", "image_path", "annotated_image"]:
        p = getattr(fp, path_attr, None)
        if p and Path(p).exists():
            try:
                Path(p).unlink()
            except OSError:
                pass

    deleted = FloorPlanDAO.delete(project_id)
    if deleted:
        return jsonify({"success": True,
                        "message": f"Project '{project_id}' deleted."}), 200
    return jsonify({"success": False,
                    "message": "Delete failed."}), 500


# ══════════════════════════════════════════════════════════
# CONTROLLER 5 — Client Share View (via JWT token)
# ══════════════════════════════════════════════════════════
@floor_plan_bp.route("/share/<string:token>", methods=["GET"])
def client_share_view(token: str):
    """
    GET /api/floor-plan/share/<jwt_token>
    Public endpoint — no login required.
    Client opens the shared link and gets plan details.

    ── Postman ──────────────────────────────────────────────
    Method : GET
    URL    : http://localhost:5000/api/floor-plan/share/<token>
    (token returned from /upload as 'share_token')
    ─────────────────────────────────────────────────────────
    """
    try:
        payload = jwt.decode(
            token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM]
        )
        project_id = payload.get("project_id")
    except jwt.ExpiredSignatureError:
        return jsonify({"success": False,
                        "message": "Share link has expired."}), 401
    except jwt.InvalidTokenError:
        return jsonify({"success": False,
                        "message": "Invalid share token."}), 401

    fp = FloorPlanDAO.get_by_id(project_id)
    if not fp:
        return jsonify({"success": False,
                        "message": "Floor plan not found."}), 404

    if fp.status != "ready":
        return jsonify({"success": False,
                        "message": f"Plan is not ready yet (status={fp.status})."}), 202

    detections = FloorPlanDAO.get_detections(project_id)
    ocr = FloorPlanDAO.get_ocr(project_id)

    plan_dict = fp.to_dict()
    plan_dict.pop("file_path", None)
    plan_dict.pop("image_path", None)

    return jsonify({
        "success": True,
        "data": {
            **plan_dict,
            "detections": [d.to_dict() for d in detections],
            "ocr": ocr.to_dict() if ocr else None,
        }
    }), 200