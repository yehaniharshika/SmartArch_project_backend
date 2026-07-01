import re

class ForgotPasswordDTO:
    def __init__(self, data: dict):
        self.email = (data.get("email", "") or "").strip().lower()

    def validate(self):
        if not self.email:
            return "Email is required."
        email_pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, self.email):
            return "Please enter a valid email address."
        return None