"""
SmartArch — services/FloorPlan_service.py
MAIN PIPELINE ORCHESTRATOR (Core Extraction Only — AI/RAG disabled for now)

STEP 1  → Validate the upload request
STEP 2  → Save the file to disk
STEP 3  → Create the database row (status="processing")
STEP 4  → Load the image (convert PDF→PNG if needed)
STEP 5  → yolo_service          → detect walls/doors/windows
STEP 6  → gemini_ocr_service    → read all text (Gemini Vision)
STEP 7  → scale                 → fixed default (scale_service disabled)
STEP 8  → room_boundary_service → find enclosed room areas from walls+doors+windows
STEP 9  → room_parser_service   → LABEL-FIRST: build rooms via boundary containment
STEP 10 → area_service          → compute final width/height/area per room
STEP 11 → Draw the annotated image
STEP 12 → Save everything to the database
STEP 13 → Generate the JWT share token (the chat link)
"""
import os
import cv2
import uuid
import time
import traceback
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import Config
from dao.FloorPlan_dao import FloorPlanDAO
from dto.AnalysisResultDTO import AnalysisResultDTO
from dto.FloorPlanUploadDTO import UploadFloorPlanRequestDTO
from dto.DetectionDTO import DetectionDTO
from dto.OCRDataDTO import OCRDataDTO

import services.extraction.yolo_service as yolo_service
import services.extraction.gemini_ocr_service as gemini_ocr_service
# from services.extraction import scale_service
import services.extraction.room_boundary_service as room_boundary_service
import services.extraction.room_parser_service as room_parser_service
import services.extraction.area_service as area_service


class FloorPlanService:

    @staticmethod
    def upload_and_analyze(user_id: int, project_name: str, file) -> tuple:
        start_time = time.time()

        error = UploadFloorPlanRequestDTO.validate(project_name)
        if error:
            return {"success": False, "message": error}, 400

        filename = file.filename
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in Config.ALLOWED_EXT:
            return {
                "success": False,
                "message": f"Unsupported file type '.{ext}'. Allowed: {Config.ALLOWED_EXT}"
            }, 400

        project_id = "PRJ-" + uuid.uuid4().hex[:6].upper()
        safe_name = f"{project_id}.{ext}"
        file_path = str(Config.UPLOAD_DIR / safe_name)
        file.save(file_path)

        print(f"\n{'='*60}")
        print(f"  NEW UPLOAD — Project: {project_id}")
        print(f"  File: {filename} → {safe_name}")
        print(f"{'='*60}")

        FloorPlanDAO.create_floor_plan(
            project_id, user_id, project_name, filename, file_path
        )

        try:
            result = FloorPlanService._run_pipeline(
                project_id, project_name, file_path, ext, start_time
            )
        except Exception as exc:
            print(f"[FATAL ERROR] Pipeline crashed:\n{traceback.format_exc()}")
            FloorPlanDAO.mark_error(project_id, str(exc))
            return {"success": False, "message": f"Analysis failed: {exc}"}, 500

        token, expires_at = FloorPlanService._make_share_token(project_id)
        FloorPlanDAO.save_share_token(project_id, token, expires_at)

        response_data = result.to_dict()
        response_data.update({
            "project_id": project_id,
            "project_name": project_name,
            "share_token": token,
            "share_url": f"/client/{token}",
        })

        FloorPlanService._print_terminal_report(result, project_name)
        return {"success": True, "data": response_data}, 200

    @staticmethod
    def _run_pipeline(project_id, project_name, file_path, ext, start_time):
        warnings = []

        # STEP 4: Load image
        img, image_path = FloorPlanService._load_image(project_id, file_path, ext)
        height, width = img.shape[:2]
        print(f"[IMAGE] Loaded {image_path} → {width}x{height}px")

        # STEP 5: YOLO detection
        try:
            raw_detections = yolo_service.detect_structural_elements(img)
        except Exception as e:
            print(f"[ERROR] YOLO detection failed: {e}")
            warnings.append(f"YOLO detection failed: {e}")
            raw_detections = []

        # STEP 6: Text extraction — Gemini Vision only.
        try:
            ocr_data = gemini_ocr_service.extract_text_gemini(img)
            ocr_engine_used = "gemini_vision"
        except Exception as e:
            print(f"[ERROR] Gemini Vision extraction failed: {e}")
            warnings.append(f"Gemini Vision extraction failed: {e}")
            ocr_data = OCRDataDTO()
            ocr_engine_used = "none"

        # STEP 7: Scale — scale_service disabled, use a fixed default.
        # Doesn't affect OCR-matched rooms (most rooms get their size
        # directly from matched dimension text); only affects the rare
        # zero-OCR-dim + real-boundary "wall_geometry_estimate" fallback.
        pixels_per_foot = 15.0
        pixels_per_meter = 49.2
        scale_method = "scale_service_disabled"
        scale_confidence = 0.0

        # STEP 8: Wall-boundary detection (used for containment-based
        # dimension matching AND as a fallback room source in room_parser)
        try:
            room_boundaries = room_boundary_service.find_room_boundaries(
                img, raw_detections
            )
            if not room_boundaries:
                warnings.append(
                    "No wall-boundary regions could be formed. Dimension "
                    "matching will fall back to radius search for every "
                    "room, which is less reliable on compact layouts."
                )
        except Exception as e:
            print(f"[ERROR] Room boundary detection failed: {e}")
            warnings.append(f"Room boundary detection failed: {e}")
            room_boundaries = []

        # STEP 9: LABEL-FIRST room building, using boundary containment
        # to match each room's label to ONLY the dimension texts inside
        # its own wall-enclosed region (prevents stealing a neighbouring
        # room's dimension on compact layouts).
        try:
            rooms = room_parser_service.build_room_objects(room_boundaries, ocr_data)
            if not rooms:
                warnings.append(
                    "No rooms could be identified. Check that the floor plan "
                    "image has clearly readable room labels."
                )
        except Exception as e:
            print(f"[ERROR] Room parsing failed: {e}")
            warnings.append(f"Room parsing failed: {e}")
            rooms = []

        # STEP 10: Calculate dimensions per room
        try:
            for room in rooms:
                area_service.calculate_room_dimensions(room, pixels_per_foot)
            total_sqft, total_sqm = area_service.calculate_total_area(rooms)
        except Exception as e:
            print(f"[ERROR] Area calculation failed: {e}")
            warnings.append(f"Area calculation failed: {e}")
            total_sqft, total_sqm = 0.0, 0.0

        detections = FloorPlanService._build_detection_dtos(
            raw_detections, pixels_per_foot, ocr_data
        )

        summary = ""

        # STEP 11: Annotated image
        try:
            annotated_path = FloorPlanService._draw_annotations(
                img.copy(), detections, rooms, project_id
            )
        except Exception as e:
            print(f"[ERROR] Drawing annotations failed: {e}")
            warnings.append(f"Could not draw annotated image: {e}")
            annotated_path = image_path

        counts = FloorPlanService._count_by_class(raw_detections)
        processing_time = round(time.time() - start_time, 2)

        result = AnalysisResultDTO(
            project_id=project_id,
            image_path=image_path,
            annotated_image=annotated_path,
            detections=detections,
            ocr_data=ocr_data,
            rooms=rooms,
            total_area_sqm=total_sqm,
            total_area_sqft=total_sqft,
            room_count=len(rooms),
            door_count=counts["door"],
            window_count=counts["window"],
            wall_count=counts["wall"],
            summary=summary,
            image_width_px=width,
            image_height_px=height,
            pixels_per_meter=pixels_per_meter,
            pixels_per_foot=pixels_per_foot,
            scale_method=scale_method,
            scale_confidence=scale_confidence,
            processing_time=processing_time,
            pipeline_warnings=warnings,
        )

        # STEP 12: Save to DB
        try:
            FloorPlanDAO.save_analysis_results(result)
            print(f"[DB] ✅ Saved project {project_id} to database "
                  f"({len(rooms)} rooms, {len(detections)} detections)")
        except Exception as e:
            print(f"[ERROR] Database save failed: {e}")
            warnings.append(f"Database save failed: {e}")

        print(f"[OCR-ENGINE] Used: {ocr_engine_used}")
        print("[RAG] ⏭️  Skipped (AI/RAG pipeline temporarily disabled — core extraction only)")
        return result

    @staticmethod
    def _load_image(project_id, file_path, ext):
        if ext == "pdf":
            import fitz
            doc = fitz.open(file_path)
            matrix = fitz.Matrix(Config.PDF_DPI / 72, Config.PDF_DPI / 72)
            pix = doc[0].get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3)
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            out_path = str(Config.UPLOAD_DIR / f"{project_id}_converted.png")
            cv2.imwrite(out_path, img)
            doc.close()
            print(f"[PDF] Converted → {Path(out_path).name}")
            return img, out_path
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not read image file: {file_path}")
        return img, file_path

    @staticmethod
    def _build_detection_dtos(raw_detections, pixels_per_foot, ocr_data):
        from services.extraction.area_service import decimal_feet_to_ft_in
        detections = []
        for det in raw_detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            pixel_w, pixel_h = x2 - x1, y2 - y1
            width_ft  = round(pixel_w / pixels_per_foot, 2) if pixels_per_foot else 0
            height_ft = round(pixel_h / pixels_per_foot, 2) if pixels_per_foot else 0
            width_m   = round(width_ft  * 0.3048, 3)
            height_m  = round(height_ft * 0.3048, 3)
            area_sqft = round(width_ft * height_ft, 3)
            area_sqm  = round(area_sqft * 0.092903, 3)
            perimeter_m = round(2 * (width_m + height_m), 3)
            matched_label = matched_dimension = None
            for t in ocr_data.raw_texts:
                cx, cy = t["center_x"], t["center_y"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    if any(ch.isdigit() for ch in t["text"]):
                        matched_dimension = t["text"]
                    else:
                        matched_label = t["text"]
            detections.append(DetectionDTO(
                label=det["label"], confidence=det["confidence"],
                x1=x1, y1=y1, x2=x2, y2=y2,
                width_m=width_m, height_m=height_m,
                width_ft_in=decimal_feet_to_ft_in(width_ft),
                height_ft_in=decimal_feet_to_ft_in(height_ft),
                area_sqm=area_sqm, area_sqft=area_sqft,
                perimeter_m=perimeter_m,
                ocr_label=matched_label,
                ocr_dimension=matched_dimension,
            ))
        return detections

    @staticmethod
    def _draw_annotations(img, detections, rooms, project_id):
        for d in detections:
            meta = Config.CLASS_META.get(d.label, {})
            hex_color = meta.get("color", "#888888").lstrip("#")
            color = (int(hex_color[4:6], 16), int(hex_color[2:4], 16), int(hex_color[0:2], 16))
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
            cv2.putText(img, f"{d.label} {d.confidence*100:.0f}%",
                       (int(d.x1)+4, int(d.y1)-6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        for room in rooms:
            cv2.rectangle(img,
                (int(room.bbox_x1), int(room.bbox_y1)),
                (int(room.bbox_x2), int(room.bbox_y2)),
                (0, 200, 255), 2)
            label_text = f"{room.name} {room.width_ft_in}x{room.height_ft_in}"
            cv2.putText(img, label_text,
                       (int(room.bbox_x1)+4, int(room.bbox_y1)+18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

        out_path = str(Config.UPLOAD_DIR / f"{project_id}_annotated.jpg")
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[ANNOTATED] Saved → {Path(out_path).name}")
        return out_path

    @staticmethod
    def _count_by_class(raw_detections):
        return {
            "door":   sum(1 for d in raw_detections if d["label"] == "door"),
            "window": sum(1 for d in raw_detections if d["label"] == "window"),
            "wall":   sum(1 for d in raw_detections if d["label"] == "wall"),
        }

    @staticmethod
    def _make_share_token(project_id):
        import jwt
        expires_at = datetime.now(timezone.utc) + timedelta(days=Config.JWT_EXPIRE_DAYS)
        token = jwt.encode(
            {"project_id": project_id, "exp": expires_at.timestamp()},
            Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM,
        )
        return token, expires_at

    @staticmethod
    def _print_terminal_report(result, project_name):
        G="\033[92m"; Y="\033[93m"; C="\033[96m"; P="\033[95m"
        BL="\033[94m"; GR="\033[90m"; R_="\033[91m"; RESET="\033[0m"; B="\033[1m"

        print(f"\n{C}{B}{'='*64}{RESET}")
        print(f"{C}{B}  SmartArch — Analysis Complete: {project_name}{RESET}")
        print(f"{C}{B}{'='*64}{RESET}")
        print(f"  {GR}{'Total area':<26}{RESET}{G}{result.total_area_sqft} sq.ft ({result.total_area_sqm} m²){RESET}")
        print(f"  {GR}{'Image size':<26}{RESET}{result.image_width_px}x{result.image_height_px}px")
        print(f"  {GR}{'Scale':<26}{RESET}{result.pixels_per_foot:.2f} px/ft ({result.scale_method}, confidence={result.scale_confidence:.2f})")
        print(f"  {GR}{'Processing time':<26}{RESET}{result.processing_time}s")
        print(f"\n{BL}{B}  STRUCTURAL ELEMENTS{RESET}")
        print(f"  Walls:{result.wall_count}  Doors:{result.door_count}  Windows:{result.window_count}")
        print(f"\n{P}{B}  ROOMS ({result.room_count} found){RESET}")
        if result.rooms:
            print(f"  {GR}{'Room':<22}{'Width':>10}{'Height':>10}{'Sq.Ft':>10}{'Source':>32}{RESET}")
            print(f"  {GR}{'-'*84}{RESET}")
            for room in result.rooms:
                src = room.dimension_source
                print(f"  {room.name:<22}{room.width_ft_in:>10}{room.height_ft_in:>10}"
                      f"{str(room.area_sqft):>10}{src:>32}")
        else:
            print(f"  {Y}No rooms identified — see warnings.{RESET}")
        if result.pipeline_warnings:
            print(f"\n{R_}{B}  ⚠ WARNINGS{RESET}")
            for w in result.pipeline_warnings:
                print(f"  {Y}• {w}{RESET}")
        print(f"\n{G}{B}{'='*64}{RESET}\n")