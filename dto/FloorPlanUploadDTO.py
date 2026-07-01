from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class UploadFloorPlanRequestDTO:
    """
    Validated from multipart/form-data:
      - project_name (form field)
      - file         (file upload: PNG / JPG / JPEG / PDF)
    """
    project_name: str
    user_id:      int

    @staticmethod
    def validate(project_name: str) -> Optional[str]:
        if not project_name or not project_name.strip():
            return "Project name is required."
        if len(project_name.strip()) > 255:
            return "Project name is too long (max 255 characters)."
        return None




