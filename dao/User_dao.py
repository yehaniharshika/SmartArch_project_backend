from entity.User_entity import User
from db import db

class UserDAO:
    @staticmethod
    def get_user_by_email(email):
        return User.query.filter_by(email=email.strip().lower()).first()
    
    @staticmethod
    def create_user(full_name, email, role, password_hash):
        new_user = User(
            full_name=full_name, 
            email=email, 
            role=role, 
            password_hash=password_hash
        )
        db.session.add(new_user)
        db.session.commit()
        return new_user

    @staticmethod
    def save_user():
        db.session.commit()

    @staticmethod
    def get_user_by_id(user_id):
        return User.query.get(user_id)
    
    @staticmethod
    def email_exists(email):
        return User.query.filter_by(email=email.strip().lower()).count() > 0
    
    @staticmethod
    def get_by_reset_token(token: str):
        return User.query.filter_by(reset_token=token).first()