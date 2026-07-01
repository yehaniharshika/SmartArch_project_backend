import json
from datetime import datetime

from db.database import db
from entity.FloorPlan_entity import FloorPlan
from entity.Detection_entity import Detection
from entity.OcrResult_entity import OCRResult
from entity.ChatMessage_entity import ChatMessage
from dto.AnalysisResultDTO import AnalysisResultDTO

class FloorPlanDAO:

    # Create floor plan
    @staticmethod
    def create_floor_plan(project_id: str, user_id: int, project_name: str,
                          filename: str, file_path: str) -> FloorPlan:
        """Insert a new FloorPlan row with status='processing'."""
        fp = FloorPlan(
            id                = project_id,
            user_id           = user_id,
            project_name      = project_name,
            original_filename = filename,
            file_path         = file_path,
            status            = "processing",
        )
        db.session.add(fp)
        db.session.commit()
        return fp

    # ── Save full analysis result ───────────────────────────
    @staticmethod
    def save_analysis_results(result: AnalysisResultDTO) -> FloorPlan | None:
        """
        Update the FloorPlan row with all extracted data.
        Also inserts one Detection row per detection and one OCRResult row.
        """
        fp = db.session.get(FloorPlan, result.project_id)
        if not fp:
            return None

        # Update FloorPlan columns
        fp.image_path        = result.image_path
        fp.annotated_image   = result.annotated_image
        fp.total_area_sqm    = result.total_area_sqm
        fp.total_area_sqft   = result.total_area_sqft
        fp.room_count        = result.room_count
        fp.door_count        = result.door_count
        fp.window_count      = result.window_count
        fp.wall_count        = result.wall_count
        fp.total_detections  = len(result.detections)
        fp.summary           = result.summary
        fp.image_width_px    = result.image_width_px
        fp.image_height_px   = result.image_height_px
        fp.pixels_per_meter  = result.pixels_per_meter
        fp.scale_method      = result.scale_method
        fp.scale_confidence  = result.scale_confidence
        fp.processing_time   = result.processing_time
        fp.status            = "ready"
        fp.updated_at        = datetime.utcnow()

        # Delete old detections (re-analyze case)
        Detection.query.filter_by(project_id=result.project_id).delete()

        # Insert each detection
        for det in result.detections:
            detection = Detection(
                project_id    = result.project_id,
                label         = det.label,
                confidence    = det.confidence,
                x1            = det.x1,
                y1            = det.y1,
                x2            = det.x2,
                y2            = det.y2,
                width_m       = det.width_m,
                height_m      = det.height_m,
                area_sqm      = det.area_sqm,
                area_sqft     = det.area_sqft,
                perimeter_m   = det.perimeter_m,
                ocr_label     = det.ocr_label,
                ocr_dimension = det.ocr_dimension,
            )
            db.session.add(detection)

        # Save OCR result
        if result.ocr_data:
            OCRResult.query.filter_by(project_id=result.project_id).delete()
            
            # 💥 FIX: Serialize python lists/dicts to JSON strings for SQLite compatibility
            ocr_row = OCRResult(
                project_id  = result.project_id,
                room_labels = json.dumps(result.ocr_data.room_labels),
                dimensions  = json.dumps(result.ocr_data.dimensions),
                raw_texts   = json.dumps(result.ocr_data.raw_texts),
            )
            db.session.add(ocr_row)

        db.session.commit()
        return fp

    # Mark error
    @staticmethod
    def mark_error(project_id: str, error_message: str) -> None:
        fp = db.session.get(FloorPlan, project_id)
        if fp:
            fp.status        = "error"
            fp.error_message = error_message
            fp.updated_at    = datetime.utcnow()
            db.session.commit()

    # Read 
    @staticmethod
    def get_by_id(project_id: str) -> FloorPlan | None:
        return db.session.get(FloorPlan, project_id)

    @staticmethod
    def get_by_user(user_id: int) -> list[FloorPlan]:
        return (FloorPlan.query
                .filter_by(user_id=user_id)
                .order_by(FloorPlan.created_at.desc())
                .all())

    @staticmethod
    def get_detections(project_id: str) -> list[Detection]:
        return (Detection.query
                .filter_by(project_id=project_id)
                .order_by(Detection.id)
                .all())

    @staticmethod
    def get_ocr(project_id: str) -> OCRResult | None:
        return OCRResult.query.filter_by(project_id=project_id).first()

    # Share token management
    @staticmethod
    def save_share_token(project_id: str, token: str,
                         expires_at: datetime) -> FloorPlan | None:
        fp = db.session.get(FloorPlan, project_id)
        if fp:
            fp.share_token      = token
            fp.share_expires_at = expires_at
            fp.updated_at       = datetime.utcnow()
            db.session.commit()
        return fp

    @staticmethod
    def get_by_share_token(token: str) -> FloorPlan | None:
        return FloorPlan.query.filter_by(share_token=token).first()

    # Delete a floor plan and all related data (detections, OCR, chat messages)
    @staticmethod
    def delete(project_id: str) -> bool:
        fp = db.session.get(FloorPlan, project_id)
        if not fp:
            return False
        db.session.delete(fp)
        db.session.commit()
        return True

    # Chat message part
    @staticmethod
    def save_chat_message(project_id: str, query: str, answer: str,
                          language: str = "en",
                          model_used: str = "gpt-4o") -> ChatMessage:
        msg = ChatMessage(
            project_id = project_id,
            query      = query,
            answer     = answer,
            language   = language,
            model_used = model_used,
        )
        db.session.add(msg)
        db.session.commit()
        return msg

    @staticmethod
    def get_chat_history(project_id: str) -> list[ChatMessage]:
        return (ChatMessage.query
                .filter_by(project_id=project_id)
                .order_by(ChatMessage.created_at.asc())
                .all())