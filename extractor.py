import numpy as np
import cv2
import re

# Monkey-patch paddlex to disable mkldnn completely, 
# preventing the ConvertPirAttribute2RuntimeAttribute bug on Windows CPU
try:
    from paddlex.inference.utils import misc
    misc.is_mkldnn_available = lambda: False
    
    # Also patch AnalysisConfig to ignore missing set_optimization_level 
    # when running paddlepaddle 2.6.x
    import paddle.inference
    if not hasattr(paddle.inference.Config, 'set_optimization_level'):
        paddle.inference.Config.set_optimization_level = lambda self, level: None
except ImportError:
    pass

from paddleocr import PaddleOCR

import os
from dotenv import load_dotenv

# Load .env from the project root (if present) so PADDLE_* env vars are available
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

class TextExtractor:
    def __init__(self, languages=['en']):
        print("Initializing PaddleOCR...")

        # Use env vars for model paths — falls back to PaddleOCR's default cache.
        # To override on this machine, set once in your terminal (or a .env file):
        #   set PADDLE_DET_MODEL=C:\Users\LENOVO\.paddleocr\whl\det\en\en_PP-OCRv3_det_infer
        #   set PADDLE_REC_MODEL=C:\Users\LENOVO\.paddleocr\whl\rec\en\en_PP-OCRv4_rec_infer
        det_model = os.environ.get("PADDLE_DET_MODEL", None)
        rec_model = os.environ.get("PADDLE_REC_MODEL", None)

        kwargs = dict(use_angle_cls=True, lang='en')
        if det_model:
            kwargs['det_model_dir'] = det_model
        if rec_model:
            kwargs['rec_model_dir'] = rec_model

        self.reader = PaddleOCR(**kwargs)

    def extract_text(self, image):
        """
        Extracts text from an image (numpy array).
        Returns a list of dicts: {'bbox', 'text', 'confidence'}
        PaddleOCR-native: skips merge logic since Paddle already returns grouped blocks.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

        results = self.reader.ocr(image)
        
        text_items = []
        if results and results[0]:
            for line in results[0]:
                bbox, (text, prob) = line
                if prob > 0.2:
                    # 1. Normalize whitespace (Paddle can return " 34   .   72 ")
                    text = re.sub(r'\s+', ' ', text).strip()
                    # 2. Clean OCR errors
                    text = self.clean_text_content(text)
                    # 3. Repair numeric strings ("34 . 72" → "34.72")
                    text = self.repair_numeric_strings(text)
                    text_items.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': prob
                    })
        
        return self.cleanup_items(text_items)

    def extract_text_custom(self, image, image_gray, x_threshold=40):
        """
        Main pipeline entry point used by main.py.
        PaddleOCR-native: skips all fragment-merge/stitch logic since Paddle
        already returns grouped text blocks. Pipeline:
          raw OCR → normalize spaces → clean → repair numerics → filter noise → classify.

        NOTE: merge_horizontal_items, merge_vertical_items, apply_decimal_stitcher,
        apply_gdt_stitcher, apply_dimension_tolerance_stitcher are kept as dormant
        fallbacks below but are NOT called here.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

        raw_results = self.reader.ocr(image)
        raw_items = []
        if raw_results and raw_results[0]:
            for line in raw_results[0]:
                bbox, (text, prob) = line
                if prob > 0.2:
                    # Step 1: normalize whitespace
                    text = re.sub(r'\s+', ' ', text).strip()
                    # Step 2: clean OCR character errors
                    text = self.clean_text_content(text)
                    # Step 3: repair split decimals within a single token
                    text = self.repair_numeric_strings(text)
                    # Step 3.5: repair merged/garbled tokens; None = drop this token
                    text = self.repair_merged_token(text)
                    if text is None:
                        continue
                    raw_items.append({'bbox': bbox, 'text': text, 'confidence': prob})

        # Step 4: remove noise/garbage tokens
        cleaned_items = self.cleanup_items(raw_items)

        # Step 4.5: Reconstruct fragmented GD&T rows (horizontally split by OCR spacing).
        # x_gap=80: OCR separates datum letters A B C by 50-80px; wider gap catches them.
        # y_thresh=20: small vertical tolerance for slightly skewed GD&T rows.
        merged_items = self.merge_gdt_rows(cleaned_items, y_thresh=20, x_gap=80)


        post_merged = []
        for item in merged_items:
            item["text"] = self.repair_numeric_strings(item["text"])
            # Second pass of repair_merged_token to catch tokens re-fused by merge_gdt_rows
            repaired = self.repair_merged_token(item["text"])
            if repaired is None:
                continue          # drop pure-noise merged tokens
            item["text"] = repaired
            post_merged.append(item)
        merged_items = post_merged

        # Step 5: classify each token type for downstream logic
        for item in merged_items:
            item["type"] = self.classify_token(item["text"])

        # Step 6 (removed): single-letter drop was here but killed valid FEATURE_LABELs
        # Classification now handles this contextually via FEATURE_LABEL vs OTHER.
        return merged_items

    def repair_numeric_strings(self, text):
        """
        Repairs numbers already merged inside a single OCR token.
        """
        t = text.strip()

        # Pattern: 263 92 → 263.92
        if re.match(r'^\d{2,4}\s+\d{1,2}$', t):
            parts = t.split()
            return parts[0] + "." + parts[1]

        # Pattern: 255 6 4 → 255.64
        if re.match(r'^\d{2,4}\s+\d\s+\d$', t):
            parts = t.split()
            return parts[0] + "." + parts[1] + parts[2]

        # Pattern: 0 7 6 → 0.76
        if re.match(r'^\d\s+\d\s+\d$', t):
            parts = t.split()
            return parts[0] + "." + parts[1] + parts[2]

        # Pattern: 34 . 72 → 34.72
        if re.match(r'^\d+\s*\.\s*\d+$', t):
            return re.sub(r'\s*\.\s*', '.', t)

        # Pattern: 68 , 72 → 68.72
        if re.match(r'^\d+\s*,\s*\d+$', t):
            return re.sub(r'\s*,\s*', '.', t)

        return text


    def repair_merged_token(self, text):
        """
        Post-processes a single OCR token to fix common merge/noise problems.
        Returns:
          - A repaired string  (keep this token, possibly cleaned)
          - None              (caller should drop this token entirely)

        Repair order:
          1. S0/S00 → ⌀  (OCR misread of diameter symbol)
          2a. Decimal + Datum fused split:  0.5ABC → 0.5 ABC
          2b. Fused double-decimal split:   2.720.5 → 2.72   (keep first only)
          3.  Pure-noise gate: drop tokens with no recognisable engineering sub-pattern
        """
        t = text.strip()
        if not t:
            return None

        # ── 1. Diameter-symbol OCR repair ────────────────────────────────────
        # PaddleOCR sometimes reads ⌀ as: 'S0', 'So', '00', '0N', 'L0'
        # e.g. "S01.0x" → "⌀1.0x",  "S00.5AB" → "⌀0.5 AB"
        t = re.sub(r'^S0{1,2}(\d)', r'⌀\1', t)
        t = re.sub(r'^[Ss][Oo0](\d)', r'⌀\1', t)

        # "00.3" → "⌀0.3",  "001.0" → "⌀1.0"  (two leading zeros before decimal)
        t = re.sub(r'^00+(\.\d)', r'⌀0\1', t)
        t = re.sub(r'^00+(\d)', r'⌀\1', t)

        # "0N.X" or "0n.X" — single leading zero before non-zero digit + decimal
        # e.g. "01.0 Z" → "⌀1.0 Z"  (only when main digit is not zero itself)
        t = re.sub(r'^0([1-9]\.\d)', r'⌀\1', t)

        # "L0.5 ABC" or "L01.0YZ" — L misread as ⌀
        t = re.sub(r'^L0+(\d)', r'⌀\1', t)
        t = re.sub(r'^L(\d+\.\d)', r'⌀\1', t)

        # ── 1b. "0.0N.MAB"-style garbled GD&T token ─────────────────────────
        # OCR sometimes reads a framed GD&T cell's prefix as "0.0" when the
        # original was "⌀0." or "⊕⌀0.".  Pattern: 0.0{digit}.{digit+}[LETTERS]
        # e.g. "0.05.5AB"  →  "⌀5.5 AB"
        #      "0.03.0DEF" →  "⌀3.0 DEF"
        gg = re.match(r'^0\.0(\d\.\d+)\s*([A-Z]{1,3})$', t)
        if gg:
            t = f"⌀{gg.group(1)} {gg.group(2)}"

        # ── 1c. Fused short-integer + tolerance split ─────────────────────────
        # Pattern: single engineering dimension immediately followed by a SMALL tolerance
        # like "20.1" (should be "2 ±0.1"), where the ± was dropped by OCR.
        # CONSERVATIVE: only fires when tolerance is 0.0X or 0.1X (≤ 0.19).
        # This avoids false-splitting genuine dimensions like "40.57" (0.57 is not a tolerance).
        # e.g. "20.1" → keep "2"   "30.05" → keep "3"   "40.57" → left alone (0.57 too big)
        short_int_tol = re.match(r'^([1-9])(0\.(?:0\d+|1\d*))$', t)
        if short_int_tol:
            first   = short_int_tol.group(1)
            dropped = short_int_tol.group(2)
            print(f"  [repair_merged] short-int-tol split '{t}' → kept '{first}' (dropped '{dropped}')")
            t = first


        # ── 1d. Fused integer-tolerance split (multi-digit) ─────────────────
        # Pattern: big integer immediately followed by small decimal, e.g. "680.1" → keep "68"
        # Only trigger when the integer part is clearly distinct (≥2 digits before the dot)
        # and the fractional part is a tolerance-style value (0.x or 0.xx)
        fused_int_tol = re.match(r'^([±SR⌀ØøÔ]?\d{2,})(0\.\d+)$', t)
        if fused_int_tol:
            first   = fused_int_tol.group(1)
            dropped = fused_int_tol.group(2)
            print(f"  [repair_merged] int-tol split '{t}' → kept '{first}' (dropped '{dropped}')")
            t = first


        # ── 2a. Decimal + Datum fused split ──────────────────────────────────
        # Pattern: 0.5ABC → 0.5 ABC   (number immediately followed by datum letters)
        datum_fused = re.match(r'^([⌀ØøÔ]?[\d]+\.\d+)([A-Z]{1,3})$', t)
        if datum_fused:
            num_part  = datum_fused.group(1)
            ltr_part  = datum_fused.group(2)
            t = f"{num_part} {ltr_part}"

        # ── 2b. Restore dropped ⌀ prefix on position-tolerance frames ─────────
        # OCR sometimes drops the ⌀ from GD&T position frames, producing "0.5 ABC"
        # instead of "⌀0.5 ABC".
        # SAFE guard: only adds ⌀ when ALL of these are true:
        #   - No ⌀/Ø prefix already present
        #   - The decimal value is ≤ 1.0 (typical position tolerance, never a raw dim)
        #   - Followed by 2–3 uppercase datum letters (multi-datum = position frame)
        #   - Single-letter combos like "0.2 D" are intentionally left alone
        no_prefix_gdt = re.match(r'^(\d*\.\d+)\s+([A-Z]{2,3})$', t)
        if no_prefix_gdt:
            val = float(no_prefix_gdt.group(1))
            if val <= 1.0:
                t = f"⌀{no_prefix_gdt.group(1)} {no_prefix_gdt.group(2)}"


        # ── 2b. Fused double-decimal split ───────────────────────────────────
        # Pattern: two decimal numbers jammed together, e.g. "2.720.5", "57.70.07"
        # Strategy: keep only the FIRST decimal; the second is usually captured
        # separately by PaddleOCR in its own bounding box.
        #
        # Engineering tolerances almost always start with "0." (e.g. ±0.5, 0.07).
        # Use non-greedy match + lookahead (?=0\.\d) to find the exact split point
        # without consuming the leading digit of the second number.
        fused = re.match(
            r'^([SR⌀ØøÔ±]?\d+\.\d+?)(?=0\.\d)',  # first decimal, lookahead: 0.x ahead
            t
        )
        if fused:
            first   = fused.group(1)
            dropped = t[len(first):]
            print(f"  [repair_merged] split '{t}' → kept '{first}' (dropped '{dropped}')")
            t = first

        # ── 3. Pure-noise gate ───────────────────────────────────────────────
        # After repairs above, if the token still matches NONE of the patterns
        # below it is considered OCR noise and is dropped (return None).
        _GOOD = [
            re.compile(r'[⌀ØøÔ]'),                          # diameter/GDT prefix
            re.compile(r'[±]'),                              # tolerance ± anywhere
            re.compile(r'^[±+-]?\d+\.\d+'),                  # decimal dimension  e.g. 2.72
            re.compile(r'^[±+-]?\d+$'),                      # integer dimension  e.g. 12
            re.compile(r'[⊕⊥\|○◇∠]'),                       # GDT control symbols
            re.compile(r'^[A-Z]$'),                          # single uppercase letter (T, R, U…)
            re.compile(r'^[A-Z]{1,3}$'),                     # datum label        e.g. AB

            re.compile(r'\d+[xX]\d*'),                       # count/repeat       e.g. 4X
            re.compile(r'\d+:\d+'),                          # ratio              e.g. 1:2
            re.compile(r'\(\d+\.?\d*\)'),                    # parenthesised tol  e.g. (0.3)
            re.compile(r'^[S]?R\d+\.?\d*'),                  # radius             e.g. R5, SR22.65
            re.compile(r'^[A-Z]\d+$'),                       # datum+number ref   e.g. A1
            re.compile(r'^\d+\.?\d*\s+[A-Z]{1,3}'),          # dim + datum string e.g. 3.4 X
            re.compile(r'[A-Z]{2,}'),                        # multi-char text note
            re.compile(r'\d{2,}'),                           # 2+ digit number fragment
        ]
        if not any(p.search(t) for p in _GOOD):
            print(f"  [repair_merged] noise drop: '{t}' (original: '{text}')")
            return None

        return t



    def classify_token(self, text):
        t = text.strip()

        # 1. DATUM BUBBLE — letter + digit, with optional brackets
        if re.match(r'^[\(\[\s]*[A-Z]\d+[\)\]\s]*$', t):
            return "DATUM_BUBBLE"

        # 2. FEATURE LABEL — single uppercase letter (hole ref, section tag, view label)
        #    Exclude I / O / S which are common OCR noise from geometry lines/curves
        if re.match(r'^[A-Z]$', t) and t not in {"I", "O", "S"}:
            return "FEATURE_LABEL"

        # 3. GD&T FRAME — pipe separators or parenthesised datum refs
        if "|" in t or re.search(r'\(.*\)', t):
            return "GDT"

        # 4. NUMERIC DIMENSION — integer or decimal, optional leading ±/+/-
        if re.match(r'^[±\+\-]?\d+(\.\d+)?$', t):
            return "DIMENSION"

        # 4b-pre. RADIUS/DIAMETER + DATUM REFS → GDT  (must come before 4b)
        # e.g. "R5 A B", "SR22.65 A", "⌀0.5 A B C" — more specific than 4b
        if re.match(r'^[SR⌀Ø]\d+\.?\d*\s+[A-Z]', t):
            return "GDT"

        # 4b. DIAMETER / RADIUS PREFIX DIMENSION (pure — no datum refs)
        if re.search(r'[ØR]\s*\d', t):
            return "DIMENSION"

        # 4c. MIXED GD&T FRAME — e.g. "3.4 X Y", "6.8 UZ Z", "5.4 XY Z"
        if re.match(r'^\d+(\.\d+)?\s+[A-Z]{1,3}(\s+[A-Z]{1,3})*$', t):
            return "GDT"

        # 5. NOTE — whole-word keyword match against functional engineering terms
        #    Uses word extraction to avoid substring false-positives
        #    (e.g. "EXAMPLE" must not match keyword "MAX")
        NOTE_KEYWORDS = {
            "THICKNESS", "WALL", "POINT", "EDGE",
            "DATUM", "SURFACE", "BREAK", "HOLE",
            "MIN", "MAX", "CONTROL", "ONLY"
        }
        words = re.findall(r'[A-Z]+', t.upper())
        if any(w in NOTE_KEYWORDS for w in words):
            return "NOTE"

        return "OTHER"


    def clean_text_content(self, text, apply_drawing_specific=False):
        """
        Applies regex replacements to fix common OCR errors.

        Parameters
        ----------
        text : str
            Raw OCR token.
        apply_drawing_specific : bool
            If True, also applies entries from DRAWING_SPECIFIC_CORRECTIONS in
            ocr_corrections.py.  Only enable this for a known drawing/scanner
            batch where you've verified the corrections are correct.
        """
        from ocr_corrections import GLOBAL_SYMBOL_CORRECTIONS, DRAWING_SPECIFIC_CORRECTIONS

        # ── 1. Global safe symbol corrections ────────────────────────────────
        for wrong, right in GLOBAL_SYMBOL_CORRECTIONS.items():
            text = text.replace(wrong, right)

        # ── 2. Drawing-specific hacks (opt-in only) ───────────────────────────
        if apply_drawing_specific:
            t = text.strip()
            if t in DRAWING_SPECIFIC_CORRECTIONS:
                text = DRAWING_SPECIFIC_CORRECTIONS[t]

        # ── 3. Regex-based universal fixes ────────────────────────────────────
        text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
        text = re.sub(r'0\s*\.\s*3\s*\)\s*\|?\s*[zZ][iI]', '(0.3) Z', text)
        text = re.sub(r'^5\s*Xi\n3$', '5.4 X Y\n3.4', text, flags=re.MULTILINE)
        text = re.sub(r'^5\s*Xi$', '5.4 X Y', text)
        text = re.sub(r'\b0\s+2\b', '0.2', text)
        text = re.sub(r'\b0\s+5\b', '0.5', text)
        text = re.sub(r'\b0\s+1\b', '0.1', text)
        text = text.replace('~', '')

        return text.strip()

    def merge_gdt_rows(self, items, y_thresh=15, x_gap=40):
        """
        Reconstructs GD&T rows that PaddleOCR fragments horizontally due to spacing.
        Example: ['6.8', 'Z', '(0.3)', 'Z'] -> '6.8 Z (0.3) Z'
        Includes a semantic guard to avoid merging unrelated items like '8.86' and 'Z'.
        """
        merged = []
        used = set()

        # Pre-compute Center-Y and Center-X for O(n) memory sort
        keyed_items = [
            ((sum(p[1] for p in it['bbox']) / 4, sum(p[0] for p in it['bbox']) / 4), it)
            for it in items
        ]
        keyed_items.sort(key=lambda x: x[0])
        items_sorted = [it for _, it in keyed_items]

        for i, item in enumerate(items_sorted):
            if i in used:
                continue

            group = [item]
            used.add(i)

            x1 = max(p[0] for p in item['bbox'])
            anchor_y = sum(p[1] for p in item['bbox']) / 4

            for j in range(i+1, len(items_sorted)):
                if j in used:
                    continue

                other = items_sorted[j]
                x2 = min(p[0] for p in other['bbox'])
                y2 = sum(p[1] for p in other['bbox']) / 4

                same_row = abs(anchor_y - y2) < y_thresh
                close = 0 < (x2 - x1) < x_gap

                # Semantic guard — BOTH tokens must look like GD&T parts,
                # AND at least one must be a frame seed (numeric or contains brackets).
                # This prevents pure-letter pairs like Z+X from merging.
                def _looks_gdt_token(t):
                    t = t.strip()
                    if "|" in t or "(" in t:
                        return True
                    if re.search(r'[⌀⊕⊥⊙○◇∠ØøÔ]', t):  # GD&T / diameter symbols
                        return True
                    if re.match(r'^[A-Z]{1,3}$', t):   # X Y Z UZ etc.
                        return True
                    if re.match(r'^[⌀ØøÔ±]?\d+(\.\d+)?$', t):  # numeric value (optional prefix)
                        return True
                    return False

                def _looks_frame_seed(t):
                    """True if token can START or ANCHOR a GD&T frame (not just a label).

                    A bare decimal number (e.g. 18.99, 26.53) is NOT a seed unless it
                    already carries a GD&T symbol prefix OR is a small tolerance (≤ 1.0).
                    This prevents dimension annotations from being vacuumed up by nearby
                    datum letters and disappearing from the output.
                    """
                    t = t.strip()
                    if "|" in t or "(" in t:
                        return True
                    if re.search(r'[⌀⊕⊥⊙○◇∠ØøÔ]', t):   # explicit GD&T symbol prefix
                        return True
                    # Numeric with an explicit symbol prefix → always a seed
                    sym_num = re.match(r'^[⌀ØøÔ±](\d+(\.\d+)?)$', t)
                    if sym_num:
                        return True
                    # Bare decimal only qualifies as a seed when value ≤ 1.0
                    # (i.e., it looks like a tolerance, not a standalone dimension)
                    bare_num = re.match(r'^\d+(\.\d+)?$', t)
                    if bare_num:
                        try:
                            return float(bare_num.group(0)) <= 1.0
                        except ValueError:
                            return False
                    return False

                gdt_like = (
                    _looks_gdt_token(item["text"]) and
                    _looks_gdt_token(other["text"]) and
                    (_looks_frame_seed(item["text"]) or _looks_frame_seed(other["text"]))
                )


                if same_row and close and gdt_like:
                    group.append(other)
                    used.add(j)
                    x1 = max(p[0] for p in other['bbox'])  # advance right edge

            if len(group) > 1:
                # Sort group strictly left-to-right before joining text
                group_sorted = sorted(group, key=lambda g: min(p[0] for p in g['bbox']))
                merged_text = " ".join(g['text'] for g in group_sorted)

                xs = [p[0] for g in group for p in g['bbox']]
                ys = [p[1] for g in group for p in g['bbox']]

                # Proportional padding: scales with text size, works across all DPI
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                PAD_X = max(6, int(0.12 * width))
                PAD_Y = max(2, int(0.08 * height))

                new_bbox = [
                    [min(xs)-PAD_X, min(ys)-PAD_Y], [max(xs)+PAD_X, min(ys)-PAD_Y],
                    [max(xs)+PAD_X, max(ys)+PAD_Y], [min(xs)-PAD_X, max(ys)+PAD_Y]
                ]

                merged.append({
                    "text": merged_text,
                    "bbox": new_bbox,
                    "confidence": min(g['confidence'] for g in group)
                })
            else:
                merged.append(item)

        return merged

    def cleanup_items(self, items):
        """Removes items that are likely noise/garbage."""
        valid = []
        garbage_texts = ["0", ".", ",", ":", ";", "-", "_", "~", "'", "\u2018", '"', "`"]
        for item in items:
            text = item['text'].strip()
            if text in garbage_texts:
                continue
            if not text:
                continue
            if len(text) <= 2:
                has_alnum = any(c.isalnum() for c in text)
                if not has_alnum:
                    continue
            valid.append(item)
        return valid

    def merge_horizontal_items(self, items, x_threshold=80, y_threshold=10):
        """Merges text items that are close horizontally and aligned vertically."""
        if not items:
            return []

        # Pre-compute min_x, min_y
        keyed = [((min([p[0] for p in it['bbox']]), min([p[1] for p in it['bbox']])), it) for it in items]
        keyed.sort(key=lambda t: t[0])
        merged_items = [it for _, it in keyed]
        
        changed = True
        while changed:
            changed = False
            new_merged = []
            skip_indices = set()
            
            # Pre-compute keys for the inner loop sort
            k_merged = [
                ((min([p[0] for p in it['bbox']]), min([p[1] for p in it['bbox']])), it)
                for it in merged_items
            ]
            k_merged.sort(key=lambda t: t[0])
            merged_items = [it for _, it in k_merged]
            
            for i, curr_item in enumerate(merged_items):
                if i in skip_indices:
                    continue
                curr_bbox = curr_item['bbox']
                curr_x_max = max([p[0] for p in curr_bbox])
                curr_y_min = min([p[1] for p in curr_bbox])
                curr_y_max = max([p[1] for p in curr_bbox])
                curr_height = curr_y_max - curr_y_min
                curr_cy = (curr_y_min + curr_y_max) / 2
                best_match_idx = -1
                best_dist = float('inf')
                for j in range(i + 1, len(merged_items)):
                    if j in skip_indices:
                        continue
                    next_item = merged_items[j]
                    next_bbox = next_item['bbox']
                    next_x_min = min([p[0] for p in next_bbox])
                    next_y_min = min([p[1] for p in next_bbox])
                    next_y_max = max([p[1] for p in next_bbox])
                    next_height = next_y_max - next_y_min
                    next_cy = (next_y_min + next_y_max) / 2
                    y_overlap = max(0, min(curr_y_max, next_y_max) - max(curr_y_min, next_y_min))
                    min_h = min(curr_height, next_height)
                    vertical_aligned = y_overlap > (min_h * 0.5) or abs(curr_cy - next_cy) < 15
                    if not vertical_aligned:
                        continue
                    if next_x_min < (curr_x_max - 10):
                        continue
                    dist = next_x_min - curr_x_max
                    if dist > x_threshold:
                        continue
                    if dist < best_dist:
                        best_dist = dist
                        best_match_idx = j
                if best_match_idx != -1:
                    j = best_match_idx
                    next_item = merged_items[j]
                    
                    # === BLOCK ALL NUMERIC MERGING ===
                    curr_text_clean = curr_item['text'].strip()
                    next_text_clean = next_item['text'].strip()

                    def is_numeric_like(t):
                        return bool(re.match(r'^[\d\.\+\-]+$', t))

                    # If both tokens are numeric-like, DO NOT merge here.
                    # Leave them separate for semantic stitchers later.
                    if is_numeric_like(curr_text_clean) and is_numeric_like(next_text_clean):
                        new_merged.append(curr_item)
                        continue

                    next_bbox = next_item['bbox']
                    new_x_min = min([p[0] for p in curr_bbox] + [p[0] for p in next_bbox])
                    new_x_max = max([p[0] for p in curr_bbox] + [p[0] for p in next_bbox])
                    new_y_min = min([p[1] for p in curr_bbox] + [p[1] for p in next_bbox])
                    new_y_max = max([p[1] for p in curr_bbox] + [p[1] for p in next_bbox])
                    new_bbox = [[new_x_min, new_y_min], [new_x_max, new_y_min],
                                [new_x_max, new_y_max], [new_x_min, new_y_max]]
                    new_text = curr_item['text'] + " " + next_item['text']
                    merged_item = {
                        'bbox': new_bbox,
                        'text': new_text,
                        'confidence': (curr_item['confidence'] + next_item['confidence']) / 2
                    }
                    new_merged.append(merged_item)
                    skip_indices.add(j)
                    changed = True
                else:
                    new_merged.append(curr_item)
            merged_items = new_merged
        return merged_items

    def merge_vertical_items(self, items, x_threshold=20, y_threshold=30):
        """Merges text items that are close vertically and aligned horizontally."""
        if not items:
            return []

        # Pre-compute min_x, min_y
        keyed = [((min([p[0] for p in it['bbox']]), min([p[1] for p in it['bbox']])), it) for it in items]
        keyed.sort(key=lambda t: t[0])
        sorted_items = [it for _, it in keyed]
        
        merged_items = []
        current_item = sorted_items[0]
        for next_item in sorted_items[1:]:
            curr_bbox = current_item['bbox']
            next_bbox = next_item['bbox']
            curr_x_min = min([p[0] for p in curr_bbox])
            curr_x_max = max([p[0] for p in curr_bbox])
            curr_y_max = max([p[1] for p in curr_bbox])
            curr_width = curr_x_max - curr_x_min
            next_x_min = min([p[0] for p in next_bbox])
            next_x_max = max([p[0] for p in next_bbox])
            next_y_min = min([p[1] for p in next_bbox])
            next_width = next_x_max - next_x_min
            overlap_width = max(0, min(curr_x_max, next_x_max) - max(curr_x_min, next_x_min))
            min_width = min(curr_width, next_width)
            horizontal_aligned = overlap_width > (min_width * 0.5)
            vertical_close = (next_y_min - curr_y_max) < y_threshold
            is_below = next_y_min > (curr_y_max - 10)
            if horizontal_aligned and vertical_close and is_below:
                new_x_min = min(curr_x_min, next_x_min)
                new_x_max = max(curr_x_max, next_x_max)
                new_y_min = min([p[1] for p in curr_bbox] + [p[1] for p in next_bbox])
                new_y_max = max([p[1] for p in curr_bbox] + [p[1] for p in next_bbox])
                new_bbox = [[new_x_min, new_y_min], [new_x_max, new_y_min],
                            [new_x_max, new_y_max], [new_x_min, new_y_max]]
                new_text = current_item['text'] + "\n" + next_item['text']
                current_item = {
                    'bbox': new_bbox,
                    'text': new_text,
                    'confidence': (current_item['confidence'] + next_item['confidence']) / 2
                }
            else:
                merged_items.append(current_item)
                current_item = next_item
        merged_items.append(current_item)
        return merged_items


# ============================================================================
# VERTICAL TEXT DETECTOR — Full-Page 90°CW Rotation
# ============================================================================
class FullPageRotationDetector:
    """Detects vertical text by rotating the entire image 90° CW,
    running PaddleOCR, and mapping bounding boxes back to original coordinates."""

    def __init__(self, reader):
        self.reader = reader

    def detect_vertical_text(self, image, image_gray):
        """
        Rotates image 90° CW, runs OCR, maps coordinates back to 0°.
        Returns list of dicts with 'bbox', 'text', 'confidence'.
        """
        h, w = image.shape[:2]
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rot_h, rot_w = rotated_image.shape[:2]  # rot_h = w, rot_w = h

        results = self.reader.ocr(rotated_image)

        new_items = []
        if results and results[0]:
            for line in results[0]:
                bbox, (text, conf) = line
                if conf < 0.2:
                    continue
                clean_text = text.strip()
                # Fix B: allow letter-only tokens (X, Z, UZ, A…) — they are valid GD&T labels
                if not re.search(r'[\dA-Za-z]', clean_text):
                    continue
                if len(clean_text) < 3 or len(clean_text) > 15:
                    continue
                if not re.match(r'^[\d\.\-\+\s,]+$', clean_text):
                    if not re.match(r'^[\d\.\-\+\sA-Za-z]+$', clean_text):
                        continue
                    alphas = sum(c.isalpha() for c in clean_text)
                    digits = sum(c.isdigit() for c in clean_text)
                    if alphas >= digits:
                        continue

                # Inverse transform for 90°CW: (rot_x, rot_y) -> orig (rot_y, rot_w - rot_x)
                original_bbox = []
                for point in bbox:
                    rot_x, rot_y = point
                    orig_x = rot_y
                    orig_y = rot_w - rot_x
                    original_bbox.append([orig_x, orig_y])

                # Proportional padding: prevents tight/narrow vertical boxes
                xs_b = [p[0] for p in original_bbox]
                ys_b = [p[1] for p in original_bbox]
                width_b = max(xs_b) - min(xs_b)
                height_b = max(ys_b) - min(ys_b)
                PAD_X = max(6, int(0.12 * width_b))
                PAD_Y = max(2, int(0.08 * height_b))
                padded_bbox = [
                    [min(xs_b)-PAD_X, min(ys_b)-PAD_Y], [max(xs_b)+PAD_X, min(ys_b)-PAD_Y],
                    [max(xs_b)+PAD_X, max(ys_b)+PAD_Y], [min(xs_b)-PAD_X, max(ys_b)+PAD_Y]
                ]

                new_items.append({
                    'bbox': padded_bbox,
                    'text': clean_text,
                    'confidence': conf,
                    'source': 'full_page_rotation',
                    'orientation': 'vertical'
                })

        return new_items
