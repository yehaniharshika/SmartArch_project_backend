class ResetPasswordDTO:
    def __init__(self, data: dict):
        self.reset_token = (data.get("reset_token", "") or "").strip()
        self.new_password = (data.get("new_password", "") or "").strip()
        self.confirm_password = (data.get("confirm_password", "") or "").strip()

    def validate(self):
        if not self.reset_token:
            return "Reset token is required."
        if not self.new_password:
            return "New password is required."
        if len(self.new_password) < 6:
            return "New password must be at least 6 characters."
        if self.new_password != self.confirm_password:
            return "Passwords do not match."
        return None