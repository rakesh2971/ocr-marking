"""
pipeline.py — Orchestrates the full annotation pipeline for one PDF.

Each stage is a separate method so it can be tested, timed, and profiled
independently. The monolithic main() body has been dissolved here.
"""
import re
import cv2
import numpy as np

from processor import DocumentProcessor
from extractor import TextExtractor, FullPageRotationDetector
from filter import AnnotationFilter
from visualizer import Visualizer
from clustering import MorphologicalClusterer
from box_detector import BoxCharacterDetector
from vector_extractor import VectorExtractor, is_vector_page


# Re-export _dedup_vertical_items so main.py can still reference it if needed
from main import _dedup_vertical_items  # noqa: F401 (helper lives in main for now)


class AnnotationPipeline:
    """
    Orchestrates the full annotation pipeline for one PDF.
    Each stage is a separate method so it can be tested and profiled independently.
    """

    def __init__(self, args):
        self.args = args
        self._init_components()

    # ── Component initialisation ─────────────────────────────────────────────

    def _init_components(self):
        """Initialize all heavy components once (not per-page)."""
        self.processor    = DocumentProcessor()
        self.extractor    = TextExtractor()
        self.vec_extractor = VectorExtractor()
        self.annot_filter = AnnotationFilter()
        self.visualizer   = Visualizer()
        self.clusterer    = MorphologicalClusterer()
        self.box_det      = BoxCharacterDetector(self.extractor.reader, zoom=self.args.zoom)
        self.fitz_doc     = self.processor.open_doc(self.args.input)
        self.easy_reader  = self._init_easy_ocr()

    def _init_easy_ocr(self):
        """Only load EasyOCR if at least one page is raster."""
        needs_ocr = any(
            not is_vector_page(self.fitz_doc[i])
            for i in range(len(self.fitz_doc))
        )
        if needs_ocr:
            import easyocr
            print("EasyOCR initialised (raster pages detected).")
            return easyocr.Reader(['en'], gpu=False, verbose=False)
        print("All pages are vector — EasyOCR skipped.")
        return None

    # ── Top-level runner ─────────────────────────────────────────────────────

    def run(self):
        """
        Run the full pipeline as a generator yielding one page at a time.
        Yields (page_idx, annotated_image, mappings_for_this_page).
        """
        print(f"Starting pipeline on: {self.args.input}")
        
        # Use the streaming generator from processor to avoid loading all pages
        for page_idx, image in self.processor.iter_pages(self.fitz_doc, zoom=self.args.zoom):
            print(f"\n--- Processing Page {page_idx + 1} ---")
            ann_img, mappings = self.process_page(page_idx, image)
            
            # Yield so the caller (main.py) can write incrementally and free RAM
            yield page_idx, ann_img, mappings

    def process_page(self, page_idx, image):
        """Full pipeline for a single page. Returns (annotated_image, mappings)."""
        fitz_page = self.fitz_doc[page_idx]

        # Pre-compute grayscale to avoid redundant 50MB array allocations per module
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Stage 1 — extract raw text items
        items = self.stage_extract(image, image_gray, fitz_page)

        # Compute circles once — used by both stage_filter and stage_detect
        print("Detecting exclusion zones (Datums)...")
        circles = self.annot_filter.detect_black_circles(image_gray)
        print(f"Found {len(circles)} exclusion circles.")

        # Stage 2 — spatial + text-pattern filtering (circles pre-computed)
        items, excl, excl_views = self.stage_filter(image, image_gray, items, circles)

        # Stage 3 — secondary detectors (boxed chars, GD&T frames, vertical)
        items = self.stage_detect(image, image_gray, items, excl, circles)

        # Stage 4 — classify + final cleanup
        items = self.stage_classify(items)

        # Stage 5 — cluster by drawing view
        cluster_info, view_rects = self.stage_cluster(image_gray, items, excl)

        # Stage 6 — EasyOCR fallback for sparse raster clusters
        cluster_info = self.stage_fallback_ocr(
            image, image_gray, fitz_page, cluster_info, view_rects, excl, excl_views, items
        )

        # Stage 7 — draw annotation balloons
        ann_img, mappings = self.stage_visualize(image, image_gray, cluster_info, page_idx)
        return ann_img, mappings

    # ── Stages ───────────────────────────────────────────────────────────────

    def stage_extract(self, image, image_gray, fitz_page):
        """Stage 1: Extract raw text items via vector path or OCR."""
        if is_vector_page(fitz_page):
            print("[vector] Extracting text from PDF glyph stream...")
            items = self.vec_extractor.extract_page_items(fitz_page, zoom=self.args.zoom)
            items = self.vec_extractor.filter_items(items, self.extractor)
            # Merge GD&T frame tokens that were fragmented per-cell by the PDF layout
            # (e.g. [⌀0.3] [D] [E] [F] → '⌀0.3 D E F' as one item).
            # x_gap=120 handles GD&T cell borders which can be 60-100px at zoom=2.
            before = len(items)
            items = self.extractor.merge_gdt_rows(items, y_thresh=20, x_gap=120)
            print(f"[vector] Found {len(items)} items from glyph stream (merged from {before}).")
        else:
            print("[raster] Extracting text via PaddleOCR...")
            items = self.extractor.extract_text_custom(image, image_gray, x_threshold=40)
            print(f"[raster] Found {len(items)} text items.")
        return items

    def stage_filter(self, image, image_gray, raw_items, circles):
        """
        Stage 2: Remove false positives.

        Accepts pre-computed circles so detect_black_circles runs only once per page.
        Returns (valid_items, excluded_items, excluded_views).
        """
        print("Filtering text...")

        # A. Datum circles
        items, excl_datums = self.annot_filter.filter_by_circles(raw_items, circles)
        print(f"  - Removed {len(excl_datums)} items inside datum circles.")

        # B. Notes section
        items, excl_notes = self.annot_filter.filter_notes_section(items)
        print(f"  - Removed {len(excl_notes)} items from Notes section.")

        # B2. Rescue drawing items incorrectly caught by notes filter
        rescued_notes, excl_notes = self.annot_filter.rescue_from_notes_filter(
            excl_notes, raw_items
        )
        if rescued_notes:
            print(f"  - RESCUED {len(rescued_notes)} drawing items from Notes filter.")
            items += rescued_notes

        # C. Bottom-right table
        items, excl_table = self.annot_filter.filter_bottom_right_table(items, image.shape)
        print(f"  - Removed {len(excl_table)} items from Bottom-Right Table.")

        # C2. GD&T rescue (only from notes — keep datums and table definitely excluded)
        gdt_rescued = self.annot_filter.rescue_gdt_items(
            excl_notes, items, self.extractor.clean_text_content
        )
        if gdt_rescued:
            items += gdt_rescued

        # C3. Detect title block boundary dynamically (used by clusterer)
        # Only needed for vector PDFs — raster uses the table filter cutoff
        self.annot_filter.detect_title_block_boundary(items, image.shape)

        # D. View labels
        items, excl_views = self.annot_filter.filter_view_labels(items)
        print(f"  - Removed {len(excl_views)} view label items.")

        # E. Top-left sheet numbers
        items, excl_corner = self.annot_filter.filter_top_left_numbers(items, image.shape)
        print(f"  - Removed {len(excl_corner)} items from Top-Left Corner.")

        excluded = excl_datums + excl_notes + excl_table + excl_views + excl_corner

        # F. Early text-pattern cleanup (part numbers / table header fragments)
        pre = len(items)
        items = self._text_pattern_cleanup(items)
        removed = pre - len(items)
        if removed:
            print(f"  - Removed {removed} items by text-pattern cleanup.")

        print(f"Valid items: {len(items)}, Total Excluded: {len(excluded)}")
        return items, excluded, excl_views

    def stage_detect(self, image, image_gray, items, excluded, circles):
        """Stage 3: Secondary detectors — boxed chars, GD&T frames, vertical text."""

        # 3a. Boxed characters (isolated datum squares)
        print("Detecting boxed characters...")
        boxed = self.box_det.detect_boxed_characters(image, image_gray, items, excluded)
        print(f"  - Found {len(boxed)} new boxed characters.")
        boxed_valid, _ = self.annot_filter.filter_by_circles(boxed, circles)
        items += boxed_valid

        # 3b. GD&T feature control frames
        gdt = self.box_det.detect_gdt_frames(image, image_gray, items, excluded)
        print(f"  - Found {len(gdt)} new GD&T frames.")
        gdt_valid, _ = self.annot_filter.filter_by_circles(gdt, circles)
        items += gdt_valid
        print(f"Total valid items after box detection: {len(items)}")

        # 3c. Vertical text (90°CW full-page rotation)
        print("Detecting vertical text via full-page rotation...")
        vert_items = self._detect_vertical(image, image_gray, items)
        print(f"  - Found {len(vert_items)} new vertical text items.")
        vert_valid, _ = self.annot_filter.filter_by_circles(vert_items, circles)
        items += vert_valid
        print(f"Total valid items after vertical detection: {len(items)}")

        # 3d. 90°CW full-image pass for diagonal / leader-line annotations
        print("Running 90°CW full-image pass for diagonal annotations...")
        diag_items = self._detect_diagonal(image, image_gray, items)
        items += diag_items

        # 3e. GD&T singleton enforcement — tighten bounding boxes
        self._tighten_gdt_bboxes(items)

        return items

    def stage_classify(self, items):
        """Stage 4: Assign type to items that don't have one + final cleanup."""
        for item in items:
            if 'type' not in item:
                item['type'] = self.extractor.classify_token(item['text'])

        n_before = len(items)
        items = self._text_pattern_cleanup(items)
        removed = n_before - len(items)
        if removed:
            print(f"  - Universal sweep removed {removed} part-number/table items.")
        return items

    def stage_cluster(self, image_gray, items, excluded):
        """Stage 5: Group items by drawing view, return (cluster_info, view_rects)."""
        print("Grouping annotations by drawing views...")
        notes_zone = (
            self.annot_filter.notes_cutoff_x,
            self.annot_filter.notes_cutoff_y,
        )
        title_block_zone = (
            getattr(self.annot_filter, 'title_block_cutoff_x', None),
            getattr(self.annot_filter, 'title_block_cutoff_y', None),
        )
        cluster_info, _, view_rects = self.clusterer.get_clusters(
            image_gray, items, exclusion_items=excluded,
            notes_zone=notes_zone, title_block_zone=title_block_zone
        )
        print(f"Views detected: {len(cluster_info)}")
        return cluster_info, view_rects

    def stage_fallback_ocr(self, image, image_gray, fitz_page, cluster_info, view_rects,
                           excluded_items, excl_views, valid_items):
        """
        Stage 6: EasyOCR fallback for under-populated raster view clusters.
        Only runs on RASTER pages (easy_reader is None for fully-vector PDFs).
        """
        from fallback_ocr import find_missing_zones, run_fallback_ocr_on_zones

        if is_vector_page(fitz_page) or self.easy_reader is None:
            return cluster_info

        _PARTNUM = re.compile(r'^\d{8,}$')
        _VL_PROX = 150
        _excl_vl_centres = [
            (sum(p[0] for p in ex['bbox']) / 4, sum(p[1] for p in ex['bbox']) / 4)
            for ex in excl_views
        ]
        _table_guard = getattr(self.annot_filter, '_table_cutoff_y', image.shape[0])

        for vi, min_x_v, centroid_y_v, cluster_items in cluster_info:
            numeric_count = sum(
                1 for it in cluster_items
                if it.get('type') in {'DIMENSION', 'GDT'}
            )
            rect = view_rects.get(vi)
            if rect is None:
                continue
            rx, ry, rw, rh = rect

            if ry + rh * 0.5 > _table_guard:
                continue

            area = rw * rh
            density = numeric_count / max(area, 1)

            # Strategy: 
            # - Always run EasyOCR on LARGE view regions (e.g. FRONT VIEW) regardless of
            #   annotation count — they can have vertical/rotated text PaddleOCR misses.
            # - For small/medium clusters: keep original skip logic so populated section
            #   views don't get flooded with fallback noise.
            is_large_view = area > 500_000  # ~700×700px bounding box
            if not is_large_view and numeric_count >= 3 and density >= 0.000002:
                continue

            view_crop = image[ry:ry + rh, rx:rx + rw]
            view_gray = image_gray[ry:ry + rh, rx:rx + rw]

            zones = find_missing_zones(view_gray, cluster_items, rx, ry)
            if not zones:
                continue

            new_items = run_fallback_ocr_on_zones(image, zones, self.easy_reader, self.extractor)
            if not new_items:
                continue

            new_items = [it for it in new_items if not _PARTNUM.match(it.get('text', '').strip())]

            # ── Quality gate for EasyOCR fallback items ──────────────────────
            # EasyOCR is less precise than PaddleOCR; apply stricter rules:
            # 1. Minimum confidence 0.65
            # 2. No non-engineering characters (& | Tz etc.)
            # 3. Not a single-char/single-digit fragment (these are split OCR artefacts)
            # 4. Must survive repair_merged_token (same bar as primary OCR)
            import re as _re
            _JUNK_CHARS = _re.compile(r'[&|\\@#$%~`^{};<>]|Tz')
            _SINGLE_DIGIT = _re.compile(r'^\d$')
            quality_items = []
            for it in new_items:
                txt = it.get('text', '').strip()
                conf = it.get('confidence', 0)
                # Reject low confidence
                if conf < 0.65:
                    print(f"  [fallback-drop] low-conf: '{txt}' ({conf:.2f})")
                    continue
                # Reject junk characters
                if _JUNK_CHARS.search(txt):
                    print(f"  [fallback-drop] junk chars: '{txt}'")
                    continue
                # Reject single digit/char fragments (pure noise)
                if _SINGLE_DIGIT.match(txt):
                    print(f"  [fallback-drop] single digit: '{txt}'")
                    continue
                # Must pass the same repair/noise gate as primary OCR
                repaired = self.extractor.repair_merged_token(txt)
                if repaired is None:
                    print(f"  [fallback-drop] noise gate: '{txt}'")
                    continue
                it['text'] = repaired
                quality_items.append(it)
            new_items = quality_items
            # ── End quality gate ──────────────────────────────────────────────

            # Drop items too close to already-excluded view labels
            filtered = []
            for it in new_items:
                cx = sum(p[0] for p in it['bbox']) / 4
                cy = sum(p[1] for p in it['bbox']) / 4
                near = any(
                    abs(cx - ecx) < _VL_PROX and abs(cy - ecy) < _VL_PROX
                    for ecx, ecy in _excl_vl_centres
                )
                if near:
                    print(f"  - Fallback OCR proximity drop: '{it['text']}' @ ({int(cx)},{int(cy)}) near view label")
                else:
                    filtered.append(it)
            new_items = filtered

            # ── Dedup against existing PaddleOCR cluster items ────────────────
            # Drop rescued items that are spatially close to an already-existing
            # annotation, or share the same text value within a wider radius.
            # This prevents EasyOCR duplicating e.g. '0.3' on top of '⌀0.30'.
            _DEDUP_POS  = 150   # px — same position  → definite duplicate
            _DEDUP_TEXT = 300   # px — same text value → probable duplicate
            existing_snap = [
                (sum(p[0] for p in ex['bbox']) / 4,
                 sum(p[1] for p in ex['bbox']) / 4,
                 ex.get('text', '').strip())
                for ex in cluster_items          # cluster_items = PaddleOCR items so far
            ]
            deduped = []
            for it in new_items:
                cx = sum(p[0] for p in it['bbox']) / 4
                cy = sum(p[1] for p in it['bbox']) / 4
                txt = it.get('text', '').strip()
                is_dup = any(
                    (abs(cx - ecx) < _DEDUP_POS  and abs(cy - ecy) < _DEDUP_POS) or
                    (abs(cx - ecx) < _DEDUP_TEXT and abs(cy - ecy) < _DEDUP_TEXT
                     and txt == etxt)
                    for ecx, ecy, etxt in existing_snap
                )
                if is_dup:
                    print(f"  [fallback-drop] dedup existing: '{txt}' @ ({int(cx)},{int(cy)})")
                else:
                    deduped.append(it)
            new_items = deduped
            # ── End dedup ─────────────────────────────────────────────────────

            if not new_items:
                continue



            cluster_items.extend(new_items)
            cluster_items.sort(key=lambda it: (
                min(p[1] for p in it['bbox']),
                min(p[0] for p in it['bbox'])
            ))
            valid_items.extend(new_items)
            print(f"  Fallback OCR rescued {len(new_items)} item(s) in view {vi}")

        return cluster_info

    def stage_visualize(self, image, image_gray, cluster_info, page_idx):
        """Stage 7: Draw annotation balloons, collect mappings."""
        annotated_img = image.copy()
        all_mappings = []
        page_counter = 1

        for _, _, _, cluster_items in cluster_info:
            annotated_img, page_mappings = self.visualizer.draw_annotations(
                annotated_img, cluster_items, start_id=page_counter, gray_image=image_gray
            )
            for m in page_mappings:
                m['page'] = page_idx + 1
            all_mappings.extend(page_mappings)
            # Advance counter only by DRAWN items (not all cluster_items)
            page_counter += len(page_mappings)

        return annotated_img, all_mappings

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _text_pattern_cleanup(items):
        """Remove 8+ digit part numbers and known table header fragments."""
        PART_NUM   = re.compile(r'^\d{8,}$')
        TABLE_HDRS = {'TABLE NO', 'ITEMPART NO', 'ITEM PART NO', 'DESCRIPTION'}
        return [
            it for it in items
            if not PART_NUM.match(it.get('text', '').strip())
            and it.get('text', '').strip().upper() not in TABLE_HDRS
        ]

    def _detect_vertical(self, image, image_gray, valid_items):
        """Vertical text detection + deduplication."""
        from main import _dedup_vertical_items, _apply_horizontal_removals

        vert_det = FullPageRotationDetector(self.extractor.reader)
        vert_items = vert_det.detect_vertical_text(image, image_gray)

        repaired = []
        for v in vert_items:
            fixed = self.extractor.repair_merged_token(v['text'])
            if fixed:
                v['text'] = fixed
                repaired.append(v)

        deduped, to_remove = _dedup_vertical_items(repaired, valid_items, image.shape)
        # Use identity-based removal to avoid value-equality collisions
        # (e.g. two items both reading "2.5" at different positions)
        new_valid = _apply_horizontal_removals(valid_items, to_remove)
        valid_items[:] = new_valid  # mutate in-place so callers see the change
        return deduped

    def _detect_diagonal(self, image, image_gray, valid_items):
        """
        90°CW full-image OCR pass to catch annotations on diagonal / leader lines.
        Returns newly found items (already filtered, not added to valid_items here).
        """
        h_img, w_img = image.shape[:2]
        
        # Rotate the pre-computed grayscale image instead of color -> color_rot -> cvtColor
        rot90_gray = cv2.rotate(image_gray, cv2.ROTATE_90_CLOCKWISE)

        # Prevent OpenCV SHRT_MAX error on very large images
        fx = fy = 2.0
        if max(rot90_gray.shape) * 2 > 30000:
            fx = fy = 1.0
            print("  - Very large image: disabling 2x upscale for 90°CW pass.")

        rot90_up = cv2.resize(rot90_gray, None, fx=fx, fy=fy,
                              interpolation=cv2.INTER_LANCZOS4)
        rot90_results = self.extractor.reader.ocr(rot90_up)

        new_items = []
        if not (rot90_results and rot90_results[0]):
            return new_items

        for line in rot90_results[0]:
            b_r, (t_r, p_r) = line
            if p_r < 0.25 or not t_r.strip():
                continue
            t_clean = self.extractor.clean_text_content(t_r.strip())
            t_clean = self.extractor.repair_numeric_strings(t_clean)
            t_clean = self.extractor.repair_merged_token(t_clean)
            if t_clean is None:
                continue
            if not re.search(r'\d+', t_clean) or len(t_clean) > 12:
                continue
            if sum(c.isdigit() for c in t_clean) < 1:
                continue
            if '=' in t_clean:
                continue
            t_clean = t_clean.split(':')[0].strip()
            if not t_clean:
                continue

            b_cx_r = sum(pt[0] for pt in b_r) / 4 / fx
            b_cy_r = sum(pt[1] for pt in b_r) / 4 / fy
            ax = int(b_cy_r)
            ay = int((h_img - 1) - b_cx_r)
            sz_r = 45
            t_digits = re.sub(r'[^\d]', '', t_clean)

            already = any(
                (
                    abs(sum(p[0] for p in ex['bbox']) / 4 - ax) < 100 and
                    abs(sum(p[1] for p in ex['bbox']) / 4 - ay) < 100
                ) or (
                    len(ex['text'].strip()) <= 10 and
                    re.sub(r'[^\d]', '', ex['text']) == t_digits and
                    len(t_digits) >= 2
                )
                for ex in valid_items
            )
            if not already:
                new_bbox = [
                    [ax - sz_r, ay - sz_r], [ax + sz_r, ay - sz_r],
                    [ax + sz_r, ay + sz_r], [ax - sz_r, ay + sz_r]
                ]
                new_items.append({'bbox': new_bbox, 'text': t_clean, 'confidence': p_r})
                print(f"  - 90°CW pass rescued: '{t_clean}' @ ({ax},{ay})")

        return new_items

    @staticmethod
    def _tighten_gdt_bboxes(items):
        """GD&T singleton enforcement — shrink bboxes so each forms its own cluster."""
        gdt_pattern = re.compile(
            r'^[|\u22a5\u2016\u25cb\u25c7\u2220\u2295]?\s*\d+[.,]\d+\s*[A-Z]?\s*$'
        )
        for item in items:
            if gdt_pattern.match(item['text'].strip()):
                cx = int(sum(p[0] for p in item['bbox']) / 4)
                cy = int(sum(p[1] for p in item['bbox']) / 4)
                hw = max(80, abs(item['bbox'][1][0] - item['bbox'][0][0]) // 2)
                hh = max(20, abs(item['bbox'][2][1] - item['bbox'][1][1]) // 2)
                item['bbox'] = [
                    [cx - hw, cy - hh], [cx + hw, cy - hh],
                    [cx + hw, cy + hh], [cx - hw, cy + hh]
                ]
