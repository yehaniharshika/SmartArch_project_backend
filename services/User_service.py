"""
SmartArch ── services/User_service.py
SERVICE LAYER for authentication.

register_user → creates account, returns JWT immediately
login_user    → verifies password, returns JWT (this is the Bearer token
                 used for floor plan upload and all other protected routes)
"""
import re
import bcrypt

from dao.User_dao import UserDAO
from utils.auth_utils import generate_user_token


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class UserService:

    # ── Register ─────────────────────────────────────────
    @staticmethod
    def register_user(data: dict) -> tuple[dict, int]:
        full_name = (data.get("full_name") or "").strip()
        email     = (data.get("email") or "").strip().lower()
        password  = data.get("password") or ""
        role      = (data.get("role") or "architect").strip()

        # Validation
        if not full_name:
            return {"success": False, "message": "Full name is required."}, 400
        if not email or not EMAIL_RE.match(email):
            return {"success": False, "message": "A valid email is required."}, 400
        if not password or len(password) < 6:
            return {"success": False,
                    "message": "Password must be at least 6 characters."}, 400

        if UserDAO.email_exists(email):
            return {"success": False,
                    "message": "An account with this email already exists."}, 409

        # Hash password
        password_hash = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        user = UserDAO.create_user(full_name, email, role, password_hash)

        # Generate JWT immediately — same token format used for login
        token = generate_user_token(user.id, user.email)

        print(f"[AUTH] Registered user_id={user.id} email={user.email}")

        return {
            "success": True,
            "message": "Registration successful.",
            "data": {
                "user": {
                    "id"       : user.id,
                    "full_name": user.full_name,
                    "email"    : user.email,
                    "role"     : user.role,
                },
                "token": token,
            }
        }, 201

    # ── Login ────────────────────────────────────────────
    @staticmethod
    def login_user(data: dict) -> tuple[dict, int]:
        email    = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        if not email or not password:
            return {"success": False,
                    "message": "Email and password are required."}, 400

        user = UserDAO.get_user_by_email(email)
        if not user:
            return {"success": False,
                    "message": "Invalid email or password."}, 401

        valid = bcrypt.checkpw(
            password.encode("utf-8"), user.password_hash.encode("utf-8")
        )
        if not valid:
            return {"success": False,
                    "message": "Invalid email or password."}, 401

        # ── This is the Bearer token used for floor plan upload ──
        token = generate_user_token(user.id, user.email)

        print(f"[AUTH] Login success user_id={user.id} email={user.email}")

        return {
            "success": True,
            "message": "Login successful.",
            "data": {
                "user": {
                    "id"       : user.id,
                    "full_name": user.full_name,
                    "email"    : user.email,
                    "role"     : user.role,
                },
                "token": token,   # ← copy this into Postman's Bearer Token field
            }
        }, 200

    # ── Forgot password (stub — extend with email sending) ──
    @staticmethod
    def forgot_password(data: dict) -> tuple[dict, int]:
        email = (data.get("email") or "").strip().lower()
        if not email:
            return {"success": False, "message": "Email is required."}, 400

        user = UserDAO.get_user_by_email(email)
        if not user:
            # Don't reveal whether the email exists
            return {"success": True,
                    "message": "If that email exists, a reset link has been sent."}, 200

        import secrets
        reset_token      = secrets.token_urlsafe(32)
        user.reset_token = reset_token
        UserDAO.save_user()

        print(f"[AUTH] Reset token for {email}: {reset_token}")
        return {"success": True,
                "message": "If that email exists, a reset link has been sent."}, 200

    # ── Reset password ──────────────────────────────────
    @staticmethod
    def reset_password(data: dict) -> tuple[dict, int]:
        token        = data.get("token") or ""
        new_password = data.get("new_password") or ""

        if not token or not new_password or len(new_password) < 6:
            return {"success": False,
                    "message": "Token and a new password (min 6 chars) are required."}, 400

        user = UserDAO.get_by_reset_token(token)
        if not user:
            return {"success": False,
                    "message": "Invalid or expired reset token."}, 400

        user.password_hash = bcrypt.hashpw(
            new_password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")
        user.reset_token = None
        UserDAO.save_user()

        return {"success": True, "message": "Password reset successful."}, 200