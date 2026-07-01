"""
SmartArch — services/extraction/room_boundary_service.py

THIS IS THE MISSING PIECE that was causing wrong dimensions.

PROBLEM: YOLOv8 only detects individual wall SEGMENTS (small rectangles).
         It has no concept of "this group of 4 walls forms one room".

SOLUTION (this file): Use classic computer-vision morphology to find
the ENCLOSED EMPTY AREAS between walls — these enclosed areas ARE
the rooms. This is a deterministic geometry algorithm, not an AI
guess, so it's explainable and reliable.

KEY FIX (this version): doors and windows are now ALSO drawn onto the
wall mask, not just walls. A door or window opening is a gap in the
physical wall line — visually correct, but for ROOM-SEPARATION purposes
it should still act as a barrier between two rooms. Two rooms sharing
a doorway are still two separate rooms with two separate dimension
labels; if we only draw "wall" detections, the doorway gap often breaks
the loop wide enough that morphological closing can't bridge it, and
the two rooms get merged into one big contour (or the loop never closes
at all and the room leaks into the background). Treating door/window
boxes as additional mask barriers keeps every room's loop closed.

HOW IT WORKS (step by step):
  1. Take the original floor-plan image.
  2. Draw every YOLO-detected wall AND door/window onto a blank black
     mask (white). Doors/windows close the gaps walls leave at openings.
  3. "Close" any remaining small gaps using morphological closing
     (for the tiny pixel-level misalignments between adjacent detections
     that aren't full doorway-sized gaps).
  4. Find all the CONTOURS (closed white-line loops) — each contour
     that encloses a sensible-sized blank area is treated as one room.
  5. Filter out contours that are too small (noise) or too large
     (the whole building outline, not a room).

This is the same general technique used in published floor-plan
parsing research (e.g. Liu et al., "Raster-to-Vector", CVPR 2017) —
using wall masks + contour detection to recover room polygons.
"""
import cv2
import numpy as np


def find_room_boundaries(img: np.ndarray, detections: list,
                         min_room_area_px: int = 2000,
                         max_room_area_ratio: float = 0.6) -> list:
    """
    Args:
        img: the original floor plan image (used only for its dimensions)
        detections: YOLO detections — walls AND door/window boxes are all
            used to build the barrier mask (see module docstring for why)
        min_room_area_px: contours smaller than this (in pixels^2) are
            ignored as noise (e.g. a tiny gap, not a real room)
        max_room_area_ratio: contours larger than this fraction of the
            WHOLE image are ignored (that's the building outline, not
            a single room)

    Returns:
        list of dicts, one per detected room boundary:
          {"contour": np.ndarray of (x,y) points,
           "bbox": (x1, y1, x2, y2),
           "area_px2": float,
           "centroid": (cx, cy)}
    """
    height, width = img.shape[:2]
    image_area = height * width

    # Step 1: blank black canvas, same size as the floor plan
    wall_mask = np.zeros((height, width), dtype=np.uint8)

    # Step 2: draw every wall AND every door/window as a filled white
    # rectangle. Doors/windows are openings in the physical wall, but
    # for the purpose of separating ROOMS into distinct enclosed areas,
    # they must act as barriers too — otherwise the loop around a room
    # leaks open exactly where a door is, and morphological closing
    # alone often can't bridge a doorway-sized gap (a door opening is
    # much wider than a few misaligned pixels between wall segments).
    barrier_detections = [d for d in detections if d["label"] in ("wall", "door", "window")]
    for det in barrier_detections:
        x1, y1 = int(det["x1"]), int(det["y1"])
        x2, y2 = int(det["x2"]), int(det["y2"])
        cv2.rectangle(wall_mask, (x1, y1), (x2, y2), 255, thickness=-1)

    # Step 3: close small REMAINING gaps — pixel-level misalignment
    # between adjacent detections, not full doorway gaps (those are
    # already covered by drawing door/window boxes above).
    kernel = np.ones((9, 9), np.uint8)
    closed_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 4: invert -- rooms are the BLACK (empty) areas now
    room_mask = cv2.bitwise_not(closed_mask)

    # Step 5: find contours of the enclosed empty (room) areas.
    # room_mask has TWO kinds of white regions: (a) the actual enclosed
    # rooms, and (b) the background OUTSIDE the building (since our
    # wall mask doesn't reach the image border). RETR_CCOMP returns
    # both as top-level contours -- we distinguish them by area: the
    # background contour is essentially the size of the whole image,
    # real rooms are much smaller. We filter the background out below
    # using max_room_area_ratio, exactly as already intended.
    contours, hierarchy = cv2.findContours(
        room_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    rooms = []
    for i, contour in enumerate(contours):
        area_px2 = cv2.contourArea(contour)

        # Only keep TOP-LEVEL contours (no parent). A contour whose
        # parent is the background means it's just the wall outline
        # traced as a "hole" inside the background -- not a real room.
        parent_index = hierarchy[0][i][3]
        if parent_index != -1:
            continue

        if area_px2 < min_room_area_px:
            continue
        if area_px2 > image_area * max_room_area_ratio:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]

        rooms.append({
            "contour": contour,
            "bbox": (x, y, x + w, y + h),
            "area_px2": area_px2,
            "centroid": (centroid_x, centroid_y),
        })

    rooms.sort(key=lambda r: r["area_px2"], reverse=True)

    print(f"[ROOM-BOUNDARY] Barrier mask built from {len(barrier_detections)} "
          f"wall/door/window detections")
    print(f"[ROOM-BOUNDARY] Found {len(rooms)} candidate room boundaries "
          f"(after filtering noise & building outline)")

    return rooms