from db import db  

def init_db(app):
    db.init_app(app)
    with app.app_context():

        from entity.User_entity import User
        from entity.FloorPlan_entity import FloorPlan
        from entity.Detection_entity import Detection
        from entity.OcrResult_entity import OCRResult
        from entity.ProjectToken_entity import ProjectToken
        from entity.ChatMessage_entity import ChatMessage

        db.create_all()
        print("✅ Database tables initialized successfully!")