"""
SmartArch ── utils/auth_utils.py
JWT helper: token generation + @token_required decorator.

Login flow:
  1. User logs in via /api/auth/login
  2. Server generates JWT with user_id encoded inside
  3. Client stores token, sends it as: Authorization: Bearer <token>
  4. Protected routes use @token_required to decode user_id automatically
"""
from functools import wraps
from datetime import datetime, timedelta, timezone

import jwt
from flask import request, jsonify, g

from config import Config


# ══════════════════════════════════════════════════════════
# Generate a login JWT (called from User_service on login/register)
# ══════════════════════════════════════════════════════════
def generate_user_token(user_id: int, email: str = "") -> str:
    """
    Create a signed JWT for a logged-in user.
    Encodes user_id — this is what protected routes will decode.
    """
    expires_at = datetime.now(timezone.utc) + timedelta(days=Config.JWT_EXPIRE_DAYS)
    payload = {
        "user_id": user_id,
        "email"  : email,
        "iat"    : datetime.now(timezone.utc).timestamp(),
        "exp"    : expires_at.timestamp(),
    }
    return jwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)


# ══════════════════════════════════════════════════════════
# Decode + validate token → returns user_id or raises
# ══════════════════════════════════════════════════════════
def decode_user_token(token: str) -> dict:
    """
    Decode JWT and return payload dict.
    Raises jwt.InvalidTokenError / jwt.ExpiredSignatureError on failure.
    """
    return jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])


# ══════════════════════════════════════════════════════════
# Decorator — protect any route, inject g.user_id
# ══════════════════════════════════════════════════════════
def token_required(f):
    """
    Use on any route that requires a logged-in user.

    Usage:
        @floor_plan_bp.route("/upload", methods=["POST"])
        @token_required
        def upload_and_analyze():
            user_id = g.user_id   # ← available here
            ...

    Expects header:
        Authorization: Bearer <jwt_token>
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return jsonify({
                "success": False,
                "message": "Missing or invalid Authorization header. "
                           "Expected format: 'Bearer <token>'"
            }), 401

        token = auth_header.split(" ", 1)[1].strip()

        try:
            payload = decode_user_token(token)
        except jwt.ExpiredSignatureError:
            return jsonify({"success": False,
                            "message": "Token has expired. Please log in again."}), 401
        except jwt.InvalidTokenError as e:
            return jsonify({"success": False,
                            "message": f"Invalid token: {str(e)}"}), 401

        user_id = payload.get("user_id")
        if user_id is None:
            return jsonify({"success": False,
                            "message": "Token does not contain user_id."}), 401

        # Make available to the route + anything it calls
        g.user_id    = user_id
        g.user_email = payload.get("email", "")

        return f(*args, **kwargs)

    return decorated