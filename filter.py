
import cv2
import numpy as np
import re


class AnnotationFilter:
    def __init__(self):
        pass

    def detect_black_circles(self, image):
        """Detects black datum circles in the image. Returns list of (x, y, r)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=10, maxRadius=40
        )
        detected_circles = []
        if circles is not None:
            detected_circles = np.uint16(np.around(circles))
            detected_circles = detected_circles[0, :]
        return detected_circles

    def is_inside_circle(self, bbox, circles):
        """Returns True if the bbox centre lies inside any circle."""
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        for (cx, cy, r) in circles:
            dist = np.sqrt((center_x - cx) ** 2 + (center_y - cy) ** 2)
            if dist < r:
                return True
        return False

    def filter_by_circles(self, text_items, circles):
        """Filters out text items inside detected datum circles."""
        valid_items = []
        excluded_items = []
        if circles is None or len(circles) == 0:
            return text_items, []
        for item in text_items:
            if self.is_inside_circle(item['bbox'], circles):
                excluded_items.append(item)
            else:
                valid_items.append(item)
        return valid_items, excluded_items

    def filter_notes_section(self, text_items):
        """Dynamically finds the NOTES header and excludes text in that column."""
        notes_header = None
        for item in text_items:
            if item['text'].strip().upper() in ["NOTES", "NOTE"]:
                notes_header = item
                break
        if not notes_header:
            print("Warning: 'NOTES' header not found. Skipping notes filtering.")
            return text_items, []

        bbox = notes_header['bbox']
        header_x_min = min([p[0] for p in bbox])
        cutoff_x = header_x_min - 20

        valid_items = []
        excluded_items = []
        for item in text_items:
            item_bbox = item['bbox']
            item_center_x = sum(p[0] for p in item_bbox) / len(item_bbox)
            if item_center_x >= cutoff_x:
                excluded_items.append(item)
            else:
                valid_items.append(item)

        return valid_items, excluded_items

    def rescue_from_notes_filter(self, excluded_notes, all_raw_items):
        """
        Notes Rescue Pass: re-admits items whose left edge is clearly in the
        drawing area (more than 200px left of the NOTES header).
        Also rescues GD&T frames and datum reference letters that are
        in the drawing but near the notes boundary.
        Returns (rescued_list, still_excluded_list).
        """
        notes_header_x = None
        for ni in all_raw_items:
            if ni['text'].strip().upper() in ['NOTES', 'NOTE']:
                notes_header_x = min(p[0] for p in ni['bbox'])
                break
        if notes_header_x is None:
            return [], excluded_notes

        rescue_cutoff = notes_header_x - 200
        
        # Pattern for GD&T / datum items worth rescuing even if close to notes boundary
        gdt_frame_pattern = re.compile(
            r'[\u2295\u2300\u00f8\u25cb\u25c7\u22a5\u2016\+\u2220|\xf8]|'
            r'^\d+\.\d+\s*[A-Z]?\s*[A-Z]?\s*[A-Z]?$'
        )
        
        rescued = []
        still_excluded = []
        for ex_item in excluded_notes:
            item_x_min = min(p[0] for p in ex_item['bbox'])
            txt = ex_item['text'].strip()
            
            # Rescue if clearly in drawing area (well left of notes)
            if item_x_min < rescue_cutoff:
                rescued.append(ex_item)
                continue
                
            # Also rescue GD&T frames / short datum letters even near notes boundary
            if gdt_frame_pattern.search(txt) or (len(txt) <= 3 and txt.isalpha() and txt.isupper()):
                rescued.append(ex_item)
                continue
                
            still_excluded.append(ex_item)
        return rescued, still_excluded

    def rescue_gdt_items(self, all_excluded, existing_valid, clean_text_fn=None):
        """
        Generic GD&T Low-Confidence Rescue:
        Re-admits any filtered item whose normalised text matches a GD&T pattern:
        - Simple dimension: X.XX[optional-letter]
        - GD&T frame with Ø and datum refs: e.g. ⊕ Ø0.5 A B C
        - Datum reference: single / double uppercase letter(s) in a box
        Dedup: skips if an existing valid item is already within 100px.
        """
        # Simple dimension
        simple_dim = re.compile(r'^\d+\.\d+\s*[A-Z]?$')
        # GD&T frame: optional symbol, Ø or ⌀, number, optional datum letters
        gdt_frame = re.compile(
            r'[\u2295\u2300\u00f8\u25cb\u25c7\u22a5\u2016\+\u2220|\xf8Oo\u00d8]'
            r'.*\d+[.,]\d+'
        )
        # Short datum-letter box: 1-3 uppercase letters only
        datum_ltr = re.compile(r'^[A-Z]{1,3}$')
        
        gdt_rescued = []
        for ex_item in all_excluded:
            raw_t = ex_item['text'].strip()
            norm = raw_t.replace('|E', ' D').replace('|e', ' D').replace('|d', ' D')
            norm = re.sub(r'(\d)\s+\.\s*(\d)', r'\1.\2', norm)
            if clean_text_fn:
                norm = clean_text_fn(norm).strip()

            is_match = (
                (simple_dim.match(norm) and len(norm) <= 8) or
                gdt_frame.search(raw_t) or
                datum_ltr.match(norm)
            )

            if is_match:
                ex_cx = sum(p[0] for p in ex_item['bbox']) / 4
                ex_cy = sum(p[1] for p in ex_item['bbox']) / 4
                already = any(
                    abs(sum(p[0] for p in v['bbox']) / 4 - ex_cx) < 100 and
                    abs(sum(p[1] for p in v['bbox']) / 4 - ex_cy) < 100
                    for v in existing_valid
                )
                if not already:
                    copy = dict(ex_item)
                    copy['text'] = norm
                    gdt_rescued.append(copy)
                    print(f"  - GD&T rescue: '{norm}' (was '{raw_t}')")
        return gdt_rescued

    def filter_bottom_right_table(self, text_items, image_shape):
        """Filters out text in the BOM/parts table, searching bottom 60% of page.
        
        Two-phase strategy:
          Phase 1 — find TABLE NO title (acts as bottom reference)
          Phase 2 — scan up to find column headers (DESCRIPTION, ITEM PART NO)
          Use the topmost header as the real upper cutoff.
        """
        if not text_items:
            return text_items, []
        h, w = image_shape[:2]
        # Search the bottom 60% of the page for any table-related anchor
        search_y_min = h * 0.40
        title_anchors   = ["TABLE NO", "TABLE NO."]
        header_anchors  = [
            "DESCRIPTION", "ITEM PART NO", "ITEMPART NO",
            "COATING WEIGHT", "MATERAIL", "MATERIAL", "PART NO"
        ]

        # ── Phase 1: find TABLE NO (bottom-most definitive row) ──────────
        table_title_y = None
        for item in text_items:
            text = item['text'].strip().upper()
            if not any(a in text for a in title_anchors):
                continue
            y_min = min(p[1] for p in item['bbox'])
            if y_min < search_y_min:
                continue
            if table_title_y is None or y_min > table_title_y:
                table_title_y = y_min          # bottommost title row

        if table_title_y is None:
            print("Warning: TABLE NO anchor not found. Skipping table filtering.")
            return text_items, []

        # ── Phase 2: scan up to find header row (real table top) ─────────
        # Look within 35% of page height above the title row
        scan_top = max(0, table_title_y - h * 0.35)
        table_top_y = table_title_y           # start pessimistic; improve below
        for item in text_items:
            text = item['text'].strip().upper()
            if not any(a in text for a in header_anchors):
                continue
            y_min = min(p[1] for p in item['bbox'])
            if scan_top <= y_min <= table_title_y:
                if y_min < table_top_y:
                    table_top_y = y_min        # move cutoff up

        cutoff_y = table_top_y - 20
        print(f"  [table filter] title_y={table_title_y:.0f}, top_y={table_top_y:.0f}, cutoff_y={cutoff_y:.0f}")

        valid_items = []
        excluded_items = []
        for item in text_items:
            y_min = min(p[1] for p in item['bbox'])
            if y_min >= cutoff_y:
                excluded_items.append(item)
            else:
                valid_items.append(item)
        self._table_cutoff_y = cutoff_y   # expose for fallback OCR guard in main.py
        return valid_items, excluded_items


    def filter_view_labels(self, text_items):
        """Filters view labels (SECTION, VIEW, FP56, etc.) unless they contain dimensions."""
        valid_items = []
        excluded_items = []
        keywords = ["VIEW", "SECTION", "DETAIL", "FRONT", "SIDE", "PT6", "PT3", "FP4", "FP3", "FP56"]
        dimension_pattern = re.compile(r'\d+\.\d+|X\s*Y\s*Z|X\s*Y|\bX\b|\bY\b|\bZ\b')
        for item in text_items:
            text = item['text'].upper()
            is_view_label = any(kw in text for kw in keywords)
            if is_view_label:
                has_dimensions = dimension_pattern.search(text) is not None
                if has_dimensions:
                    valid_items.append(item)
                else:
                    excluded_items.append(item)
            else:
                valid_items.append(item)
        return valid_items, excluded_items

    def filter_top_left_numbers(self, text_items, image_shape):
        """Filters garbage sheet numbers in the extreme top-left corner."""
        if not text_items:
            return text_items, []
        h, w = image_shape[:2]
        valid_items = []
        excluded_items = []
        for item in text_items:
            bbox = item['bbox']
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            center_x = sum(x_coords) / 4
            center_y = sum(y_coords) / 4
            if center_x < (w * 0.25) and center_y < (h * 0.035):
                excluded_items.append(item)
            else:
                valid_items.append(item)
        return valid_items, excluded_items
