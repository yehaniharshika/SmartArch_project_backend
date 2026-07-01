from flask import Blueprint, request, jsonify
from services.User_service import UserService

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    response, status_code = UserService.register_user(data)
    return jsonify(response), status_code

@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    response, status_code = UserService.login_user(data)
    return jsonify(response), status_code

@auth_bp.route("/api/auth/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json() or {}
    response, status_code = UserService.forgot_password(data)
    return jsonify(response), status_code

@auth_bp.route("/api/auth/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json() or {}
    response, status_code = UserService.reset_password(data)
    return jsonify(response), status_code