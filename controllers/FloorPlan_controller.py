import jwt
from flask import Blueprint, request, jsonify, g
from pathlib import Path

from config import Config
from dao.FloorPlan_dao import FloorPlanDAO
from utils.auth_utils import token_required
from flask import send_file

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


# Upload & Analyze Floor Plan
@floor_plan_bp.route("/upload", methods=["POST"])
@token_required
def upload_and_analyze():
    """
    POST /api/floor-plan/upload
    Headers: Authorization: Bearer <jwt_token_from_login>
    multipart/form-data:
      project_name  (text, required)
      file          (file, required: PNG/JPG/JPEG/PDF)
    """
    user_id = g.user_id

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


# Get Full Result by project_id
@floor_plan_bp.route("/<string:project_id>", methods=["GET"])
@token_required
def get_floor_plan(project_id: str):
    """
    GET /api/floor-plan/<project_id>
    Headers: Authorization: Bearer <token>
    Returns full analysis result for a project, including rooms.
    """
    fp = FloorPlanDAO.get_by_id(project_id)
    if not fp:
        return jsonify({"success": False,
                        "message": f"Project '{project_id}' not found."}), 404

    if fp.user_id != g.user_id:
        return jsonify({"success": False,
                        "message": "You do not have access to this project."}), 403

    detections = FloorPlanDAO.get_detections(project_id)
    ocr        = FloorPlanDAO.get_ocr(project_id)
    rooms      = FloorPlanDAO.get_rooms(project_id)          # ← NEW

    return jsonify({
        "success": True,
        "data": {
            **fp.to_dict(),
            "detections": [d.to_dict() for d in detections],
            "ocr":        ocr.to_dict() if ocr else None,
            "rooms":      [r.to_dict() for r in rooms],       # ← NEW
        }
    }), 200


# List All Plans for the Logged-In User
@floor_plan_bp.route("/my-plans", methods=["GET"])
@token_required
def get_user_plans():
    """
    GET /api/floor-plan/my-plans
    Headers: Authorization: Bearer <token>
    """
    plans = FloorPlanDAO.get_by_user(g.user_id)
    return jsonify({
        "success": True,
        "count": len(plans),
        "data": [fp.to_dict() for fp in plans],
    }), 200


# Delete a Floor Plan
@floor_plan_bp.route("/<string:project_id>", methods=["DELETE"])
@token_required
def delete_floor_plan(project_id: str):
    """
    DELETE /api/floor-plan/<project_id>
    Headers: Authorization: Bearer <token>
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


# Client Share View (via JWT token)
@floor_plan_bp.route("/share/<string:token>", methods=["GET"])
def client_share_view(token: str):
    """
    GET /api/floor-plan/share/<jwt_token>
    Public endpoint — no login required.
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
    ocr        = FloorPlanDAO.get_ocr(project_id)
    rooms      = FloorPlanDAO.get_rooms(project_id)          # ← NEW

    plan_dict = fp.to_dict()
    plan_dict.pop("file_path", None)
    plan_dict.pop("image_path", None)

    return jsonify({
        "success": True,
        "data": {
            **plan_dict,
            "detections": [d.to_dict() for d in detections],
            "ocr":        ocr.to_dict() if ocr else None,
            "rooms":      [r.to_dict() for r in rooms],       # ← NEW
        }
    }), 200


@floor_plan_bp.route("/<string:project_id>/annotated", methods=["GET"])
@token_required
def get_annotated_image(project_id: str):
    """
    GET /api/floor-plan/<project_id>/annotated
    Streams the YOLOv8-annotated image file (not JSON) for use in <img src>.
    """
    fp = FloorPlanDAO.get_by_id(project_id)
    if not fp:
        return jsonify({"success": False, "message": "Project not found."}), 404

    if fp.user_id != g.user_id:
        return jsonify({"success": False, "message": "Access denied."}), 403

    if not fp.annotated_image or not Path(fp.annotated_image).exists():
        return jsonify({"success": False, "message": "Annotated image not found."}), 404

    return send_file(fp.annotated_image, mimetype="image/jpeg")