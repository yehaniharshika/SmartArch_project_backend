import os
import sys
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
sys.path.insert(0, os.path.dirname(__file__))
from flask import Flask, jsonify
from flask_cors import CORS
from config import Config
from db.database import db, init_db
from controllers.User_controller import auth_bp
from controllers.FloorPlan_controller import floor_plan_bp

def create_app():
    app = Flask(__name__)

    app.config.from_object(Config)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_UPLOAD_MB * 1024 * 1024

    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000"]}}, supports_credentials=True)

    # Initialize database
    init_db(app)

    # User routes
    app.register_blueprint(auth_bp)
    app.register_blueprint(floor_plan_bp)


    with app.app_context():
        from entity.User_entity import User
        from entity.FloorPlan_entity import FloorPlan
        from entity.Detection_entity import Detection
        from entity.OcrResult_entity import OCRResult
        from entity.ProjectToken_entity import ProjectToken
        from entity.ChatMessage_entity import ChatMessage
        db.create_all()

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "message": "SmartArch backend running"
        })

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"success": False, "message": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"success": False, "message": "Internal server error"}), 500

    return app


if __name__ == "__main__":
    app = create_app()

    # Create folders
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(Config.VECTORSTORE_DIR, exist_ok=True)

    print("\n======================================")
    print(" SmartArch Backend Starting")
    print("======================================")
    print(f"Port : {Config.FLASK_PORT}")
    print(f"Env  : {Config.FLASK_ENV}")
    print("======================================\n")

    app.run(
        host="0.0.0.0",
        port=Config.FLASK_PORT,
        debug=(Config.FLASK_ENV == "development"),
        use_reloader=False
    )