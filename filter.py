
import cv2
import numpy as np
import re


class AnnotationFilter:
    def __init__(self):
        # Set by filter_notes_section; used by clusterer to dynamically
        # exclude the notes zone instead of relying on hardcoded percentages.
        self.notes_cutoff_x = None
        self.notes_cutoff_y = None
        self.title_block_cutoff_x = None
        self.title_block_cutoff_y = None

    def detect_black_circles(self, image_gray):
        """Detects filled datum circles using Hough transform + filled-contour check.
        
        Returns list of (x, y, r).  False-positive rate is reduced by:
        • Higher circularity threshold (0.83 vs 0.70)
        • Narrower radius range — datum markers are 20-65px at zoom=2
        • Filled-interior check — datum circles are solid black, not outlines
        """
        gray = image_gray
        circles_found = []

        # ── 1. HoughCircles at half resolution ─────────────────────────────
        SCALE = 0.5
        small   = cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        blurred = cv2.medianBlur(small, 5)
        hf = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1,
            minDist=int(25 * SCALE),           # wider min-gap → fewer duplicates
            param1=50, param2=32,              # slightly stricter accumulator threshold
            minRadius=int(10 * SCALE),
            maxRadius=int(35 * SCALE)          # tighter: datum circles are small
        )
        if hf is not None:
            for c in hf[0]:
                circles_found.append((c[0] / SCALE, c[1] / SCALE, c[2] / SCALE))

        # ── 2. Contour pass on full-resolution ─────────────────────────────
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = gray.shape

        for c in contours:
            area = cv2.contourArea(c)
            if not (300 < area < 15000):
                continue
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue

            # Stricter circularity: 0.83 excludes arcs, slots and angular holes
            circularity = 4 * np.pi * (area / (peri * peri))
            if circularity < 0.83:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            aspect = float(w) / h
            if not (0.8 < aspect < 1.25):
                continue

            r = max(w, h) / 2.0
            # Narrower radius range for datum circles at zoom=2
            if not (20 < r < 65):
                continue

            cx_c = x + w / 2.0
            cy_c = y + h / 2.0

            # Filled-interior check: datum markers are solid black blobs.
            # Sample pixels in the inner 60% of the circle; if the mean is bright
            # (not dark) reject this as an outline/border circle.
            mask = np.zeros(gray.shape, dtype=np.uint8)
            inner_r = max(1, int(r * 0.6))
            cv2.circle(mask, (int(cx_c), int(cy_c)), inner_r, 255, -1)
            inner_pixels = gray[mask == 255]
            if inner_pixels.size == 0 or np.mean(inner_pixels) > 160:
                # Interior is bright → outline circle, not a filled datum marker
                continue

            circles_found.append((cx_c, cy_c, r))

        # ── 3. Deduplicate (wider threshold catches near-misses from two passes) ─
        DEDUP_DIST = 30   # px — wider than 15 to merge Hough vs contour detections
        deduped: list[tuple] = []
        for cx, cy, cr in circles_found:
            best = None
            best_dist = DEDUP_DIST
            for idx, (dx, dy, dr) in enumerate(deduped):
                d = ((cx - dx) ** 2 + (cy - dy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best = idx
            if best is None:
                deduped.append((cx, cy, cr))
            else:
                dx, dy, dr = deduped[best]
                if cr > dr:            # keep larger radius
                    deduped[best] = (cx, cy, cr)

        print(f"  [circles] {len(circles_found)} raw → {len(deduped)} after dedup")
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
        """Dynamically finds the NOTES header and excludes text in that notes column.

        Uses a 2-D exclusion zone (right of AND below the header) so that
        drawing annotations on the right side of the page are NOT wrongly cut
        when the NOTES section is in the middle/lower-right area.
        """

        NOTE_ALIASES = {
            "NOTES", "NOTE", "NOTES:", "NOTE:",
            "GENERAL NOTES", "GEN. NOTES", "GEN NOTES",
            "GENERAL NOTE", "DRAWING NOTES"
        }

        notes_header = None
        for item in text_items:
            text_upper = item['text'].strip().upper()
            # Exact match OR starts-with (catches e.g. "NOTES (UNLESS OTHERWISE SPECIFIED)")
            if text_upper in NOTE_ALIASES or any(text_upper.startswith(a) for a in NOTE_ALIASES):
                notes_header = item
                break

        if not notes_header:
            print("Warning: Notes header not found. Skipping notes filtering.")
            return text_items, []

        bbox = notes_header['bbox']
        header_x_min = min([p[0] for p in bbox])
        header_y_min = min([p[1] for p in bbox])

        # Exclusion zone: items that are BOTH to the right of AND at/below the
        # NOTES header. This prevents cutting drawing content above or to the
        # left of the notes block.
        cutoff_x = header_x_min - 20  # small left-margin tolerance
        cutoff_y = header_y_min - 20  # small top-margin tolerance

        # Expose for downstream use (clusterer, fallback OCR guard, …)
        self.notes_cutoff_x = cutoff_x
        self.notes_cutoff_y = cutoff_y

        valid_items = []
        excluded_items = []
        for item in text_items:
            item_xs = [p[0] for p in item['bbox']]
            item_ys = [p[1] for p in item['bbox']]
            item_center_x = sum(item_xs) / len(item_xs)
            item_y_min    = min(item_ys)

            # Only exclude if item is within the notes column AND below the header.
            # Extra guard: if the item's LEFT edge is clearly to the left of the
            # notes header x, it's a drawing annotation — never exclude it.
            item_x_min = min(item_xs)
            in_notes_column = item_center_x >= cutoff_x and item_y_min >= cutoff_y
            clearly_left_of_notes = item_x_min < (cutoff_x - 100)
            if in_notes_column and not clearly_left_of_notes:
                excluded_items.append(item)
            else:
                valid_items.append(item)

        print(f"  [notes filter] header at x={header_x_min:.0f}, y={header_y_min:.0f}  "
              f"→ excluded {len(excluded_items)} items in notes zone")
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
        
        from box_detector import _build_spatial_index, _get_nearby
        spatial_index = _build_spatial_index(existing_valid, cell_size=200)
        
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
                
                nearby_valid = _get_nearby(spatial_index, ex_cx, ex_cy, cell_size=200)
                already = False
                for v in nearby_valid:
                    if abs(sum(p[0] for p in v['bbox']) / 4 - ex_cx) < 100 and \
                       abs(sum(p[1] for p in v['bbox']) / 4 - ex_cy) < 100:
                        already = True
                        break
                        
                if not already:
                    copy = dict(ex_item)
                    copy['text'] = norm
                    gdt_rescued.append(copy)
                    print(f"  - GD&T rescue: '{norm}' (was '{raw_t}')")
        return gdt_rescued

    def filter_bottom_right_table(self, text_items, image_shape):
        """Filters out text in the BOM/parts table, searching bottom 75% of page.
        
        Two-phase strategy:
          Phase 1 — find TABLE NO title (acts as bottom reference)
          Phase 2 — scan up to find column headers (DESCRIPTION, ITEM PART NO)
          Use the topmost header as the real upper cutoff.
        """
        if not text_items:
            return text_items, []
        h, w = image_shape[:2]
        # Search the bottom 75% of the page for any table-related anchor
        search_y_min = h * 0.25
        title_anchors = [
            "TABLE NO", "TABLE NO.", "BILL OF MATERIALS",
            "BOM", "PARTS LIST", "ITEM NO", "ITEM NO."
        ]
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
        # "FP3 SECTION VIEW C-C", "PT6 SECTION VIEW A-A", "SECTION A-A"
        section_view = re.compile(
            r'^(?:FP\d+\s+)?(?:PT\d+\s+)?SECTION(?:\s+VIEW)?\s+[A-Z]-[A-Z]$'
        )
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
        _excluded_snapshot = list(excluded_items)
        _excl_targets = [
            (min(p[0] for p in ex['bbox']), sum(p[1] for p in ex['bbox']) / 4)
            for ex in _excluded_snapshot
        ]

        swept_valid = []
        newly_excluded = []
        
        for item in valid_items:
            ci_cx = sum(p[0] for p in item['bbox']) / 4
            ci_cy = sum(p[1] for p in item['bbox']) / 4
            text  = item['text'].strip()

            # Only consider sweeping SHORT tokens (len ≤ 3) that look like stray
            # OCR fragments (e.g. "6" split off from "PT6", "B" from "B-B").
            # Full dimension strings like "18.99" or "26.53" are NEVER prefix fragments.
            is_short_fragment = len(text) <= 3 and not re.search(r'\d+\.\d+', text)

            is_prefix = is_short_fragment and any(
                abs(ci_cy - ex_cy) < 25 and 0 < ex_x_min - ci_cx < 80
                for (ex_x_min, ex_cy) in _excl_targets
            )

            
            if is_prefix:
                newly_excluded.append(item)
            else:
                swept_valid.append(item)
                
        excluded_items.extend(newly_excluded)
        valid_items = swept_valid

        return valid_items, excluded_items


    def filter_top_left_numbers(self, text_items, image_shape):
        """Filters garbage sheet reference numbers from the drawing border margins.
        
        Covers:
        - Top-left corner (sheet number, rev letter)  — top 3.5% × left 25%
        - Entire top border strip                     — top 3.5% any x
        - Entire bottom border strip                  — bottom 3.5% any x
        - Right-edge strip after the title block area — right 3% any y
        """
        if not text_items:
            return text_items, []
        h, w = image_shape[:2]
        top_y    = h * 0.035    # top border
        bottom_y = h * 0.965    # bottom border
        right_x  = w * 0.97    # right border strip

        valid_items = []
        excluded_items = []
        for item in text_items:
            bbox = item['bbox']
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            center_x = sum(x_coords) / 4
            center_y = sum(y_coords) / 4

            in_top_border    = center_y < top_y
            in_bottom_border = center_y > bottom_y
            in_right_border  = center_x > right_x

            if in_top_border or in_bottom_border or in_right_border:
                excluded_items.append(item)
            else:
                valid_items.append(item)
        return valid_items, excluded_items

    def detect_title_block_boundary(self, text_items, image_shape):
        """
        Finds the top-left corner of the title block in the bottom-right area.
        Stores result in self.title_block_cutoff_x / _y for the clusterer.
        Returns (cutoff_x, cutoff_y) or (None, None) if not found.

        WHY: The old hardcoded 65%/80% cutoff in clustering.py dropped valid
        section views on wide drawings (bumper, fascia) where drawing content
        fills the full page. This detects the real title block boundary instead.
        Safe for raster PDFs — if no anchor found, cutoffs stay None and the
        clusterer falls back gracefully with no change to raster behaviour.
        """
        h, w = image_shape[:2]

        # Only search bottom 30% AND right 40% of page
        search_x_min = w * 0.60
        search_y_min = h * 0.70

        TITLE_BLOCK_ANCHORS = [
            "CUS. PART NO", "CUS PART NO", "CUST PART NO",
            "CUS. PART NAME", "CUST. PART NAME",
            "MATE PART NO", "MATE PART NAME",
            "DRAWN BY", "CHECKED BY", "APPROVED BY",
            "DRG NO", "DRG. NO", "DRAWING NO", "DRAWING NUMBER",
            "REVISION", "SHEET NO",
            "MATHERSON", "MOTHERSON", "TATA MOTORS", "MAHINDRA",
        ]

        topmost_y = None
        leftmost_x = None

        for item in text_items:
            text = item['text'].strip().upper()
            cx = sum(p[0] for p in item['bbox']) / 4
            cy = sum(p[1] for p in item['bbox']) / 4

            # Only consider items in the bottom-right zone
            if cx < search_x_min or cy < search_y_min:
                continue

            if any(anchor in text for anchor in TITLE_BLOCK_ANCHORS):
                y_min = min(p[1] for p in item['bbox'])
                x_min = min(p[0] for p in item['bbox'])

                if topmost_y is None or y_min < topmost_y:
                    topmost_y = y_min
                if leftmost_x is None or x_min < leftmost_x:
                    leftmost_x = x_min

        if topmost_y is None:
            self.title_block_cutoff_x = None
            self.title_block_cutoff_y = None
            print("  [title block] no anchor found — full page kept for clustering")
            return None, None

        # Use a larger top margin so section views immediately above the
        # title block are not accidentally cut. The title block itself has
        # internal padding so 20px was cutting into the view above it.
        # 60px gives enough clearance for the bottom border line + labels.
        cutoff_y = max(0, topmost_y - 60)
        cutoff_x = max(0, (leftmost_x - 20) if leftmost_x else w * 0.55)
        print(f"  [title block] anchor at y={topmost_y:.0f}, x={leftmost_x:.0f} "
              f"-> cutoff y={cutoff_y:.0f}, x={cutoff_x:.0f}")

        self.title_block_cutoff_x = cutoff_x
        self.title_block_cutoff_y = cutoff_y
        return cutoff_x, cutoff_y

