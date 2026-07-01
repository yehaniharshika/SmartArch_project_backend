import re

VALID_ROLES = ["architect", "draftsman", "engineer"]

class UserRegisterDTO:
    def __init__(self, data: dict):
        self.full_name = (data.get("full_name", "") or "").strip()
        self.email = (data.get("email", "") or "").strip().lower()
        self.role = (data.get("role", "architect") or "architect").strip().lower()
        self.password = (data.get("password", "") or "").strip()
        self.confirm_password = (data.get("confirm_password", "") or "").strip()

    def validate(self):
        if not self.full_name or len(self.full_name) < 2:
            return "Full name must be at least 2 characters."
        if len(self.full_name) > 150:
            return "Full name is too long (max 150 characters)."
        if not self.email:
            return "Email is required."
        email_pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, self.email):
            return "Please enter a valid email address."
        if not self.password:
            return "Password is required."
        if len(self.password) < 6:
            return "Password must be at least 6 characters."
        if len(self.password) > 128:
            return "Password is too long."
        if self.password != self.confirm_password:
            return "Passwords do not match."
        if self.role not in VALID_ROLES:
            return f"Invalid role. Choose from: {', '.join(VALID_ROLES)}."
        return None