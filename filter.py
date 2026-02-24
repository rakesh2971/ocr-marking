
import cv2
import numpy as np
import re


class AnnotationFilter:
    def __init__(self):
        pass

    def detect_black_circles(self, image):
        """Detects black datum circles in the image using Hough transform and Contours. Returns list of (x, y, r)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        circles_found = []
            
        # 1. HoughCircles Pass (best for perfect raster circles)
        blurred = cv2.medianBlur(gray, 5)
        hf_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=10, maxRadius=40
        )
        if hf_circles is not None:
            for c in hf_circles[0]:
                circles_found.append((c[0], c[1], c[2]))
                
        # 2. Contours Pass (catches ellipses and polylines from vector PDFs)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            area = cv2.contourArea(c)
            # 300px area is roughly r=10, 20000px area is roughly r=80
            if 300 < area < 20000:
                peri = cv2.arcLength(c, True)
                if peri == 0: continue
                
                circularity = 4 * np.pi * (area / (peri * peri))
                x, y, w, h = cv2.boundingRect(c)
                
                # Protect against divide by zero
                if h == 0: continue
                aspect = float(w) / h
                
                # Permissive circularity for ellipses (aspect up to 1.4)
                if 0.6 < aspect < 1.4 and circularity > 0.4:
                    r = max(w, h) / 2.0
                    if r > 10:
                        cx = x + w / 2.0
                        cy = y + h / 2.0
                        circles_found.append((cx, cy, r))
                        
        # 3. Deduplicate
        deduped = []
        for cx, cy, cr in circles_found:
            is_dup = False
            for dx, dy, dr in deduped:
                if ((cx - dx)**2 + (cy - dy)**2)**0.5 < 15:
                    is_dup = True
                    # Keep the larger radius to be safe when excluding text
                    if cr > dr:
                        deduped.remove((dx, dy, dr))
                        deduped.append((cx, cy, cr))
                    break
            if not is_dup:
                deduped.append((cx, cy, cr))
                
        return deduped

    def is_inside_circle(self, bbox, circles):
        """Returns True if the bbox centre or its corners lie inside any circle."""
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        
        for (cx, cy, r) in circles:
            dist_center = np.sqrt((center_x - cx) ** 2 + (center_y - cy) ** 2)
            # Standard check: center is inside the circle
            if dist_center < r:
                return True
                
            # Fallback for vector text: vector bboxes are extremely tight and their 
            # geometric center might be offset. Check if the entire box is 
            # roughly contained within the circle's expanded radius.
            expanded_r = r * 1.3
            corners_inside = 0
            for px, py in zip(x_coords, y_coords):
                if np.sqrt((px - cx)**2 + (py - cy)**2) < expanded_r:
                    corners_inside += 1
                    
            if corners_inside >= 3: # If 3 or 4 corners are inside, it's inside
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
        """Filters view labels (SECTION, VIEW, FP56, B-B, etc.) unless they contain dimensions."""
        valid_items = []
        excluded_items = []
        keywords = [
            "VIEW", "SECTION", "DETAIL", "FRONT", "SIDE",
            "PT6", "PT3", "FP4", "FP3", "FP56",
            "CLEARANCE", "SCALE",
        ]

        # ── Regex patterns for view/section title forms ──────────────────────
        # Any two-letter section-cut: A-Z dash A-Z  (covers B-B, D-D, A-B, etc.)
        section_cut = re.compile(r'^[A-Z]-[A-Z]$')
        # "VIEW B", "VIEW A", "VIEW AB" (standalone letter(s) after VIEW keyword)
        view_letter = re.compile(r'^VIEW\s+[A-Z]{1,3}$')
        # "SECTION VIEW D-D", "SECTION A-A"
        section_view = re.compile(r'^SECTION(?:\s+VIEW)?\s+[A-Z]-[A-Z]$')
        # Standalone FP codes: FP3, FP56, FP123, PT3, PT6 …
        fp_code = re.compile(r'^(?:FP|PT)\d+$')
        # Phrases like "CLEARANCE HOLE DETAIL", "FP3 MIN FLAT DETAIL", "MIN FLAT DETAIL"
        detail_phrase = re.compile(
            r'(?:CLEARANCE\s+HOLE\s+DETAIL|MIN(?:IMUM)?\s+FLAT\s+DETAIL'
            r'|FP\d+\s+MIN\s+FLAT\s+DETAIL)'
        )
        # Scale lines: "SCALE 1:2", "SCALE 2:1"
        scale_line = re.compile(r'^SCALE\s+\d+:\d+$')

        dimension_pattern = re.compile(r'\d+\.\d+|X\s*Y\s*Z|X\s*Y|\bX\b|\bY\b|\bZ\b')

        for item in text_items:
            text = item['text'].strip()
            text_up = text.upper()

            # ── Hard-exclude by regex (regardless of dimension content) ──────
            if (section_cut.match(text_up)
                    or view_letter.match(text_up)
                    or section_view.match(text_up)
                    or fp_code.match(text_up)
                    or detail_phrase.search(text_up)
                    or scale_line.match(text_up)):
                excluded_items.append(item)
                continue

            # ── Soft-exclude by keyword (skip if contains a real dimension) ──
            is_view_label = any(kw in text_up for kw in keywords)
            if is_view_label:
                has_dimensions = dimension_pattern.search(text_up) is not None
                if has_dimensions:
                    valid_items.append(item)
                else:
                    excluded_items.append(item)
            else:
                valid_items.append(item)

        # ── Proximity sweep: remove valid items that are split OCR prefixes ──
        # If a valid item sits on the same text row and immediately to the LEFT
        # of an excluded view label, it's a prefix fragment (e.g. "6" from "PT6").
        # Threshold: within 25px vertically, within 250px horizontally to the left.
        swept_valid = []
        for item in valid_items:
            ci_cx = sum(p[0] for p in item['bbox']) / 4
            ci_cy = sum(p[1] for p in item['bbox']) / 4
            is_prefix = False
            for ex in excluded_items:
                ex_x_min = min(p[0] for p in ex['bbox'])
                ex_cy    = sum(p[1] for p in ex['bbox']) / 4
                if (abs(ci_cy - ex_cy) < 25                # same row
                        and 0 < ex_x_min - ci_cx < 250):   # item to the left
                    is_prefix = True
                    break
            if is_prefix:
                excluded_items.append(item)
            else:
                swept_valid.append(item)
        valid_items = swept_valid

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
