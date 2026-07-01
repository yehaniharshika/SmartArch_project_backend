class UserLoginDTO:
    def __init__(self, data: dict):
        self.email = (data.get("email", "") or "").strip().lower()
        self.password = (data.get("password", "") or "").strip()

    def validate(self):
        if not self.email:
            return "Email is required."
        if not self.password:
            return "Password is required."
        return None