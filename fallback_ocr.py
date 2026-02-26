"""
fallback_ocr.py — Targeted EasyOCR fallback for dimensions missed by PaddleOCR.

Flow:
  detect_dimension_lines(gray_crop) → finds H/V lines via HoughLinesP
  find_missing_zones(view_gray, items, rx, ry) → lines with no text nearby
  run_fallback_ocr_on_zones(image, zones, easy_reader, extractor) → new items
"""

import cv2
import numpy as np
import re


# ── 1. Dimension line detector ────────────────────────────────────────────────

def detect_dimension_lines(gray, min_line_len=80, max_gap=10):
    """
    Find near-horizontal and near-vertical lines in a grayscale crop.
    Returns list of (x1, y1, x2, y2) in crop-local coordinates.
    """
    # Auto-threshold: works for both high-contrast vector and low-contrast scans
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low  = max(0,   int(otsu_thresh * 0.5))
    high = min(255, int(otsu_thresh * 1.5))
    edges = cv2.Canny(gray, low, high)

    # Try progressively lower Hough thresholds if no lines found
    for hough_thresh in (120, 80, 50):
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=hough_thresh,
            minLineLength=min_line_len,
            maxLineGap=max_gap
        )
        if lines is not None:
            break

    result = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx > 60 and dy < 10:          # near-horizontal
                result.append((x1, y1, x2, y2))
            elif dy > 60 and dx < 10:        # near-vertical
                result.append((x1, y1, x2, y2))
    return result


# ── 2. Find lines with no text nearby ────────────────────────────────────────

def find_missing_zones(view_gray, items, rx, ry, line_pad=40):
    """
    For each dimension line in the view crop, check whether any existing item
    is within (60px H, 40px V) of the line midpoint. Lines with no nearby token
    become candidate crop zones.

    Returns list of (x_min, y_min, x_max, y_max) in PAGE coordinates.
    """
    lines = detect_dimension_lines(view_gray)
    h, w = view_gray.shape
    zones = []

    for x1, y1, x2, y2 in lines:
        cx_line = (x1 + x2) / 2
        cy_line = (y1 + y2) / 2

        has_text = False
        for it in items:
            # Convert item bbox centre to crop-local coords
            bx = sum(p[0] for p in it['bbox']) / 4 - rx
            by = sum(p[1] for p in it['bbox']) / 4 - ry
            if abs(bx - cx_line) < 60 and abs(by - cy_line) < 40:
                has_text = True
                break

        if not has_text:
            # Build padded crop zone in crop-local coords, then convert to page
            x_lo = max(0, int(min(x1, x2)) - line_pad)
            x_hi = min(w, int(max(x1, x2)) + line_pad)
            y_lo = max(0, int(min(y1, y2)) - line_pad)
            y_hi = min(h, int(max(y1, y2)) + line_pad)
            # Back to page coordinates
            zones.append((x_lo + rx, y_lo + ry, x_hi + rx, y_hi + ry))

    # Deduplicate zones that heavily overlap (keep unique)
    unique = []
    for z in zones:
        if not any(
            abs(z[0] - u[0]) < 50 and abs(z[1] - u[1]) < 50
            for u in unique
        ):
            unique.append(z)
    return unique


# ── 3. Run EasyOCR on crop zones ─────────────────────────────────────────────

_DIM_PATTERN = re.compile(r'^\d+([.,]\d+)?$')
_GDT_PATTERN = re.compile(r'[\d.,±\+\-\(\)\|]')
_PART_NUM_PATTERN = re.compile(r'^\d{8,}$')   # 8+ digits = part number, not a dimension
_SECTION_CUT_PATTERN = re.compile(r'^[A-Z]-[A-Z]$')  # B-B, A-A, D-D section labels


def _is_useful_token(text):
    """Keep numeric dimensions and GD&T-like tokens. Drop pure letters / noise / part numbers / section labels."""
    t = text.strip()
    if not t:
        return False
    if _PART_NUM_PATTERN.match(t):              # reject 8+ digit part numbers
        return False
    if _SECTION_CUT_PATTERN.match(t.upper()):   # reject section-cut labels like B-B
        return False
    if _DIM_PATTERN.match(t):
        return True
    if len(t) >= 3 and _GDT_PATTERN.search(t):
        return True
    return False


def run_fallback_ocr_on_zones(image, zones, easy_reader, extractor):
    """
    For each zone (page-coord rectangle), crop from the full image, upscale 2×,
    run EasyOCR, filter to useful tokens, convert bbox back to page coords,
    classify, and return new item dicts.
    """
    new_items = []

    for (px0, py0, px1, py1) in zones:
        # Clip to image bounds
        h, w = image.shape[:2]
        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(w, px1), min(h, py1)

        if px1 <= px0 or py1 <= py0:
            continue

        crop = image[py0:py1, px0:px1]

        # Upscale 2× to give EasyOCR more pixels
        crop_up = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

        import concurrent.futures

        def _easyocr_with_timeout(reader, crop, timeout_sec=15):
            """Run EasyOCR with a timeout. Returns [] if it times out."""
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(reader.readtext, crop, detail=1)
                try:
                    return future.result(timeout=timeout_sec)
                except concurrent.futures.TimeoutError:
                    print(f"  [fallback_ocr] EasyOCR timed out after {timeout_sec}s — skipping zone")
                    return []
                except Exception as e:
                    print(f"  [fallback_ocr] EasyOCR error: {e}")
                    return []

        results = _easyocr_with_timeout(easy_reader, crop_up, timeout_sec=15)
        
        if not results:
            continue

        for (bbox_e, text_raw, conf) in results:
            if conf < 0.3:
                continue
            text = extractor.clean_text_content(text_raw.strip())
            text = extractor.repair_numeric_strings(text)
            if not _is_useful_token(text):
                continue

            # EasyOCR bbox is [[x0,y0],[x1,y0],[x1,y1],[x0,y1]] in upscaled coords
            # Scale back to original crop, then shift to page coords
            page_bbox = [
                [int(pt[0] / 2) + px0, int(pt[1] / 2) + py0]
                for pt in bbox_e
            ]

            item = {
                'text': text,
                'bbox': page_bbox,
                'confidence': conf,
                'source': 'fallback_easyocr',
                'type': extractor.classify_token(text),
            }
            new_items.append(item)
            print(f"  [fallback_ocr] rescued: '{text}' conf={conf:.2f} @ ({px0},{py0})")

    return new_items
