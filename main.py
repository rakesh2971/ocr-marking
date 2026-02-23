import argparse
import os
import csv
import re
import cv2
import numpy as np
# Force IPv4 resolution to prevent BCEBOS download failures (IPv6 not always routable on Windows)
import socket as _socket
_orig_getaddrinfo = _socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, _socket.AF_INET, type, proto, flags)
_socket.getaddrinfo = _ipv4_getaddrinfo

# Disable PaddlePaddle PIR and MKLDNN (causes instructions crashes on Windows)
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

# Fix Paddlex API mismatch on PaddlePaddle 2.6.x (AttributeError: set_optimization_level)
try:
    import paddle.inference
    if not hasattr(paddle.inference.Config, 'set_optimization_level'):
        paddle.inference.Config.set_optimization_level = lambda self, level: None
except ImportError:
    pass

from processor import DocumentProcessor
from extractor import TextExtractor, FullPageRotationDetector
from filter import AnnotationFilter
from visualizer import Visualizer
from clustering import MorphologicalClusterer
from box_detector import BoxCharacterDetector


def main():
    parser = argparse.ArgumentParser(description="Automated Drawing Marking System")
    parser.add_argument("--input", required=True, help="Input PDF file path")
    parser.add_argument("--output_pdf", required=True, help="Output Annotated PDF file path")
    parser.add_argument("--output_csv", required=True, help="Output CSV file path")
    parser.add_argument("--zoom", type=float, default=2.0, help="Zoom factor (default: 2.0)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    # Initialize components
    processor = DocumentProcessor()
    extractor = TextExtractor()
    annot_filter = AnnotationFilter()
    visualizer = Visualizer()
    clusterer = MorphologicalClusterer()

    # Initialise EasyOCR once for the targeted fallback pass
    import easyocr as _easyocr
    from fallback_ocr import find_missing_zones, run_fallback_ocr_on_zones
    easy_reader = _easyocr.Reader(['en'], gpu=False, verbose=False)

    print(f"Processing {args.input}...")

    images = processor.pdf_to_images(args.input, zoom=args.zoom)
    print(f"pdf_to_images: Extracted {len(images)} pages.")

    annotated_images = []
    all_mappings = []

    for page_idx, image in enumerate(images):
        # Numbering is per-page: each page starts from 1 (cluster-wise, not global)
        page_counter = 1
        print(f"--- Processing Page {page_idx + 1} ---")

        # ── 1. Extract text with custom thresholds ──────────────────────────
        # Uses x_threshold=40 (tighter than default 80) to prevent GD&T frames
        # merging with adjacent dimension values.
        # Applies decimal stitcher + GD&T stitcher internally.
        print("Extracting text with custom thresholds...")
        raw_text_items = extractor.extract_text_custom(image, x_threshold=40)
        print(f"Found {len(raw_text_items)} text items (after custom filtering).")

        # ── 2. Detect exclusion zones (datum circles) ───────────────────────
        print("Detecting exclusion zones (Datums)...")
        circles = annot_filter.detect_black_circles(image)
        print(f"Found {len(circles)} exclusion circles.")

        # ── 3. Filter text ──────────────────────────────────────────────────
        print("Filtering text...")

        # A. Datum circles
        valid_items_1, excluded_datums = annot_filter.filter_by_circles(raw_text_items, circles)
        print(f"  - Removed {len(excluded_datums)} items inside datum circles.")

        # B. Notes section
        valid_items_2, excluded_notes = annot_filter.filter_notes_section(valid_items_1)
        print(f"  - Removed {len(excluded_notes)} items from Notes section.")

        # B2. Notes rescue pass: re-admit drawing items incorrectly caught by notes filter
        rescued_notes, excluded_notes = annot_filter.rescue_from_notes_filter(
            excluded_notes, raw_text_items
        )
        if rescued_notes:
            print(f"  - RESCUED {len(rescued_notes)} drawing items from Notes filter.")
            valid_items_2 = valid_items_2 + rescued_notes

        # C. Bottom-right table
        valid_items_3, excluded_table = annot_filter.filter_bottom_right_table(valid_items_2, image.shape)
        print(f"  - Removed {len(excluded_table)} items from Bottom-Right Table.")

        # C2. Generic GD&T low-confidence rescue
        # NOTE: deliberately exclude excluded_table so that table data like
        # "680.1" or "34" are not mistaken for engineering dimensions.
        all_excluded_so_far = excluded_datums + excluded_notes
        gdt_rescued = annot_filter.rescue_gdt_items(
            all_excluded_so_far,
            valid_items_3,
            clean_text_fn=extractor.clean_text_content
        )
        if gdt_rescued:
            valid_items_3 = valid_items_3 + gdt_rescued

        # D. View labels
        valid_items_4, excluded_views = annot_filter.filter_view_labels(valid_items_3)
        print(f"  - Removed {len(excluded_views)} view label items.")

        # E. Top-left sheet numbers
        valid_items, excluded_top_left = annot_filter.filter_top_left_numbers(valid_items_4, image.shape)
        print(f"  - Removed {len(excluded_top_left)} items from Top-Left Corner.")

        excluded_items = (excluded_datums + excluded_notes + excluded_table
                          + excluded_views + excluded_top_left)

        # F. Final text-pattern cleanup: drop 8+ digit part numbers and
        #    table header fragments that slip through spatial filters.
        import re as _re
        _PARTNUM = _re.compile(r'^\d{8,}$')
        _TABLE_HEADERS = {'TABLE NO', 'ITEMPART NO', 'ITEM PART NO', 'DESCRIPTION'}
        _pre_cleanup = len(valid_items)
        valid_items = [
            it for it in valid_items
            if not _PARTNUM.match(it.get('text', '').strip())
            and it.get('text', '').strip().upper() not in _TABLE_HEADERS
        ]
        print(f"  - Removed {_pre_cleanup - len(valid_items)} items by text-pattern cleanup (part numbers / table headers).")
        print(f"Valid items: {len(valid_items)}, Total Excluded: {len(excluded_items)}")

        # ── 4. Detect boxed characters ──────────────────────────────────────
        print(f"Detecting boxed characters...")
        box_detector = BoxCharacterDetector(extractor.reader)
        boxed_chars = box_detector.detect_boxed_characters(
            image,
            existing_items=valid_items,
            exclusion_items=excluded_items
        )
        print(f"  - Found {len(boxed_chars)} new boxed characters.")
        valid_items = valid_items + boxed_chars
        
        # Also detect GD&T feature control frames (multi-compartment boxes like ⊕ Ø0.5 A B C)
        gdt_frames = box_detector.detect_gdt_frames(
            image,
            existing_items=valid_items,
            exclusion_items=excluded_items
        )
        print(f"  - Found {len(gdt_frames)} new GD&T frames.")
        valid_items = valid_items + gdt_frames
        print(f"Total valid items after box detection: {len(valid_items)}")

        # ── 4.5. Vertical text pass (90°CW full-page rotation) ──────────────
        print("Detecting vertical text via full-page rotation...")
        vert_detector = FullPageRotationDetector(extractor.reader)
        vert_items = vert_detector.detect_vertical_text(image)

        deduped_vert = []
        horizontal_to_remove = []
        for v_item in vert_items:
            v_bbox = v_item['bbox']
            v_x_min = min(p[0] for p in v_bbox)
            v_x_max = max(p[0] for p in v_bbox)
            v_y_min = min(p[1] for p in v_bbox)
            v_y_max = max(p[1] for p in v_bbox)
            is_dup = False
            for h_item in valid_items:
                h_bbox = h_item['bbox']
                h_x_min = min(p[0] for p in h_bbox)
                h_x_max = max(p[0] for p in h_bbox)
                h_y_min = min(p[1] for p in h_bbox)
                h_y_max = max(p[1] for p in h_bbox)
                if not (v_x_max < h_x_min - 5 or v_x_min > h_x_max + 5 or
                        v_y_max < h_y_min - 5 or v_y_min > h_y_max + 5):
                    h_w = h_x_max - h_x_min
                    h_h = h_y_max - h_y_min
                    if h_h > h_w * 1.2:
                        horizontal_to_remove.append(h_item)
                        continue
                    h_text = h_item['text'].strip()
                    alnum_count = sum(c.isalnum() for c in h_text)
                    if alnum_count < 3 and len(h_text) <= 4:
                        horizontal_to_remove.append(h_item)
                        continue
                    is_dup = True
                    break
            h_img_sz, w_img_sz = image.shape[:2]
            v_cx = (v_x_min + v_x_max) / 2
            v_cy = (v_y_min + v_y_max) / 2
            if v_cx > w_img_sz * 0.70 and v_cy > h_img_sz * 0.80:
                is_dup = True
            if v_cx > w_img_sz * 0.72 and v_cy < h_img_sz * 0.85:
                is_dup = True
            if not is_dup:
                deduped_vert.append(v_item)

        for h_item in horizontal_to_remove:
            if h_item in valid_items:
                valid_items.remove(h_item)

        print(f"  - Found {len(deduped_vert)} new vertical text items.")
        valid_items = valid_items + deduped_vert
        print(f"Total valid items after vertical detection: {len(valid_items)}")

        # ── 4.7. 90°CW full-image pass for diagonal/leader-line annotations ─
        print("Running 90°CW full-image pass for diagonal annotations...")
        h_img, w_img = image.shape[:2]
        rot90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rot90_gray = cv2.cvtColor(rot90, cv2.COLOR_BGR2GRAY)
        rot90_up = cv2.resize(rot90_gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        rot90_results = extractor.reader.ocr(rot90_up)

        if rot90_results and rot90_results[0]:
            for line in rot90_results[0]:
                b_r, (t_r, p_r) = line
                if p_r < 0.35 or not t_r.strip():
                    continue
                t_clean = extractor.clean_text_content(t_r.strip())
                if not re.search(r'\d+[.,]\d+', t_clean) or len(t_clean) > 12:
                    continue
                if sum(c.isdigit() for c in t_clean) < 2:
                    continue
                if '=' in t_clean:
                    continue
                t_clean = t_clean.split(':')[0].strip()
                if not t_clean:
                    continue
                # Inverse transform for ROTATE_90_CLOCKWISE on (H, W) image:
                # original_x (col) = y_r / 2  (row of rotated / scale)
                # original_y (row) = (H-1) - x_r / 2  (col of rotated / scale, flipped)
                b_cx_r = sum(pt[0] for pt in b_r) / 4 / 2
                b_cy_r = sum(pt[1] for pt in b_r) / 4 / 2
                ax = int(b_cy_r)
                ay = int((h_img - 1) - b_cx_r)
                sz_r = 45
                t_digits = re.sub(r'[^\d]', '', t_clean)
                already_r = any(
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
                if not already_r:
                    new_bbox_r = [[ax - sz_r, ay - sz_r], [ax + sz_r, ay - sz_r],
                                  [ax + sz_r, ay + sz_r], [ax - sz_r, ay + sz_r]]
                    valid_items.append({'bbox': new_bbox_r, 'text': t_clean, 'confidence': p_r})
                    print(f"  - 90°CW pass rescued: '{t_clean}' @ ({ax},{ay})")

        # ── 4.8. GD&T singleton enforcement ─────────────────────────────────
        # Tighten the bbox of any GD&T frame item so it forms its own cluster.
        gdt_pattern = re.compile(
            r'^[|\u22a5\u2016\u25cb\u25c7\u2220\u2295]?\s*\d+[.,]\d+\s*[A-Z]?\s*$'
        )
        for item in valid_items:
            if gdt_pattern.match(item['text'].strip()):
                cx_g = int(sum(p[0] for p in item['bbox']) / 4)
                cy_g = int(sum(p[1] for p in item['bbox']) / 4)
                hw_g = max(80, abs(item['bbox'][1][0] - item['bbox'][0][0]) // 2)
                hh_g = max(20, abs(item['bbox'][2][1] - item['bbox'][1][1]) // 2)
                item['bbox'] = [
                    [cx_g - hw_g, cy_g - hh_g],
                    [cx_g + hw_g, cy_g - hh_g],
                    [cx_g + hw_g, cy_g + hh_g],
                    [cx_g - hw_g, cy_g + hh_g]
                ]

        # ── 4.9. (PaddleOCR-native: dimension+tolerance stitcher removed) ──────
        # PaddleOCR already returns grouped blocks; stitching is no longer needed.

        # ── 4.10. Classify any items that don't yet have a type ─────────────────
        # Items from box_detector, FullPageRotationDetector and the 90°CW pass
        # are appended directly to valid_items without a 'type' key.
        for item in valid_items:
            if 'type' not in item:
                item['type'] = extractor.classify_token(item['text'])

        # ── 4.10b. Final universal text-pattern sweep ─────────────────────────
        # Runs AFTER all item sources (PaddleOCR, box_detector, vertical, 90°CW).
        # Catches items like 548232103801 added by box_detector AFTER step F.
        import re as _re2
        _PARTNUM2 = _re2.compile(r'^\d{8,}$')
        _TABLE_HDR2 = {'TABLE NO', 'ITEMPART NO', 'ITEM PART NO', 'DESCRIPTION'}
        _n_before = len(valid_items)
        valid_items = [
            it for it in valid_items
            if not _PARTNUM2.match(it.get('text', '').strip())
            and it.get('text', '').strip().upper() not in _TABLE_HDR2
        ]
        if _n_before != len(valid_items):
            print(f"  - Universal sweep removed {_n_before - len(valid_items)} part-number/table items.")

        # ── 5. Group by drawing views + sort in reading order ─────────────────
        print("Grouping annotations by drawing views...")
        cluster_info, _, view_rects_map = clusterer.get_clusters(image, valid_items, exclusion_items=excluded_items)
        print(f"Views detected: {len(cluster_info)}")

        # ── 5.5. Targeted fallback OCR on under-populated view clusters ────────────
        # Only triggers for clusters with very few DIMENSION/GDT tokens relative to view area.
        # Runs EasyOCR only on crop zones where dimension lines have no nearby text.
        for vi, min_x_v, centroid_y_v, cluster_items in cluster_info:
            numeric_count = sum(
                1 for it in cluster_items
                if it.get('type') in {'DIMENSION', 'GDT'}
            )
            rect = view_rects_map.get(vi)
            if rect is None:
                continue
            rx, ry, rw, rh = rect

            # Skip views that overlap with the BOM/material table zone.
            # _table_cutoff_y is set by filter_bottom_right_table.
            _table_guard = getattr(annot_filter, '_table_cutoff_y', image.shape[0])
            if ry + rh * 0.5 > _table_guard:
                continue

            area = rw * rh
            density = numeric_count / max(area, 1)

            # Skip healthy clusters
            if numeric_count >= 3 and density >= 0.000002:
                continue

            # Correct colour-space conversion (image may be BGR in this pipeline)
            view_crop = image[ry:ry + rh, rx:rx + rw]
            if len(view_crop.shape) == 3:
                view_gray = cv2.cvtColor(view_crop, cv2.COLOR_BGR2GRAY)
            else:
                view_gray = view_crop

            zones = find_missing_zones(view_gray, cluster_items, rx, ry)
            if not zones:
                continue

            new_items = run_fallback_ocr_on_zones(image, zones, easy_reader, extractor)
            if new_items:
                # Strip any part numbers that slipped past _is_useful_token
                new_items = [
                    it for it in new_items
                    if not _PARTNUM2.match(it.get('text', '').strip())
                ]

                # ── View-label proximity guard ─────────────────────────────
                # Fallback OCR runs after filter_view_labels, so rescued items
                # can land next to view-title text (e.g. "6" beside SECTION VIEW A-A).
                # Drop any rescued item whose centre is within 150 px of an
                # already-excluded view-label centroid.
                _VL_PROX = 150
                _excl_vl_centres = [
                    (sum(p[0] for p in ex['bbox']) / 4,
                     sum(p[1] for p in ex['bbox']) / 4)
                    for ex in excluded_views
                ]
                _filtered_items = []
                for _it in new_items:
                    _cx = sum(p[0] for p in _it['bbox']) / 4
                    _cy = sum(p[1] for p in _it['bbox']) / 4
                    _near = any(
                        abs(_cx - _ex_cx) < _VL_PROX and abs(_cy - _ex_cy) < _VL_PROX
                        for _ex_cx, _ex_cy in _excl_vl_centres
                    )
                    if _near:
                        print(f"  - Fallback OCR proximity drop: '{_it['text']}' @ ({int(_cx)},{int(_cy)}) near view label")
                    else:
                        _filtered_items.append(_it)
                new_items = _filtered_items
                # ─────────────────────────────────────────────────────────

                if not new_items:
                    continue
                cluster_items.extend(new_items)
                # Re-sort in reading order after insertion
                cluster_items.sort(key=lambda it: (
                    min(p[1] for p in it['bbox']),
                    min(p[0] for p in it['bbox'])
                ))
                valid_items.extend(new_items)
                print(f"  Fallback OCR rescued {len(new_items)} item(s) in view {vi}")

        # ── 6. Visualize (draw per-cluster numbering) ───────────────
        annotated_img = image.copy()

        for _, _, _, cluster_items in cluster_info:

            annotated_img, page_mappings = visualizer.draw_annotations(
                annotated_img,
                cluster_items,
                start_id=page_counter
            )

            for m in page_mappings:
                m['page'] = page_idx + 1

            all_mappings.extend(page_mappings)

            # Advance counter only by DRAWN items (page_mappings), not all cluster_items.
            # Using len(cluster_items) created phantom IDs for skipped items, causing gaps.
            page_counter += len(page_mappings)

        annotated_images.append(annotated_img)

    # ── 7. Save annotated PDF ────────────────────────────────────────────────
    print(f"Saving annotated PDF to {args.output_pdf}...")
    processor.images_to_pdf(annotated_images, args.output_pdf)

    # ── 8. Save CSV (TPEM Technical Review Sheet format) ────────────────────
    # S.No. is per-page (cluster-wise); Page column identifies the sheet.
    # Export ALL drawn items — DRAW_TYPES in visualizer already controls what's drawn,
    # so every item here belongs to a meaningful annotation type.
    print(f"Saving mapping to {args.output_csv}...")
    with open(args.output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['Page', 'S.No.', 'Parameters Critical to fitment & Function']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_mappings:
            writer.writerow({
                'Page': row['page'],
                'S.No.': row['id'],
                'Parameters Critical to fitment & Function': row['description']
            })

    print("Done!")


if __name__ == "__main__":
    main()
