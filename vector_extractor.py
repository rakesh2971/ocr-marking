"""
vector_extractor.py — Direct text extraction from vector/born-digital PDFs.

Uses PyMuPDF page.get_text("words") which already groups characters into words,
avoiding the span-fragmentation problem that occurs with "dict" mode.

Output format matches the OCR pipeline exactly:
    {'bbox': [[x0,y0],[x1,y0],[x1,y1],[x0,y1]], 'text': str,
      'confidence': 1.0, 'source': 'vector', 'type': str}

So every downstream module (filter, clusterer, visualizer, CSV) works unchanged.
"""

import fitz  # PyMuPDF


# ── Detection helper ──────────────────────────────────────────────────────────

VECTOR_CHAR_THRESHOLD = 50  # chars per page to be treated as vector


def is_vector_page(page: fitz.Page, threshold: int = VECTOR_CHAR_THRESHOLD) -> bool:
    """
    Returns True if this single page has embedded selectable text (vector).
    Detection is intentionally per-page so mixed documents work correctly:
      page 1 = vector drawing → fast path
      page 2 = scanned sheet  → OCR path
    """
    text = page.get_text("text").strip()
    return len(text) >= threshold


# ── Main extractor class ──────────────────────────────────────────────────────

class VectorExtractor:
    """
    Extracts text from a vector PDF page and converts it to item dicts that
    are compatible with the rest of the annotation pipeline.
    """

    # Minimum word length to keep (skip single punctuation artifacts)
    MIN_WORD_LEN = 1

    def extract_page_items(self, page: fitz.Page, zoom: float = 2.0) -> list:
        """
        Extract all words from a PyMuPDF page and return item dicts.

        Args:
            page  : fitz.Page object (NOT the rendered image)
            zoom  : same zoom factor used when rasterising (fitz.Matrix(zoom, zoom))
                    All PDF-unit coordinates are multiplied by this value so that
                    bbox coordinates match the pixel positions in the rendered image.

        Returns:
            List of item dicts: {'bbox', 'text', 'confidence', 'source', 'type'}
        """
        # page.get_text("words") returns a list of:
        # (x0, y0, x1, y1, word, block_no, line_no, word_no)
        # Coordinates are in PDF user-space units (72 dpi baseline).
        words = page.get_text("words")

        items = []
        for entry in words:
            x0, y0, x1, y1, word_text = entry[0], entry[1], entry[2], entry[3], entry[4]
            text = word_text.strip()
            if len(text) < self.MIN_WORD_LEN:
                continue

            # Scale PDF coords → pixel coords matching the rasterised image
            px0 = x0 * zoom
            py0 = y0 * zoom
            px1 = x1 * zoom
            py1 = y1 * zoom

            # 4-corner bbox format used by filter / clusterer / visualizer
            bbox = [
                [px0, py0],  # top-left
                [px1, py0],  # top-right
                [px1, py1],  # bottom-right
                [px0, py1],  # bottom-left
            ]

            items.append({
                'bbox':       bbox,
                'text':       text,
                'confidence': 1.0,   # vector text is exact — no OCR confidence needed
                'source':     'vector',
                'type':       None,  # filled in by filter_items()
            })

        return self._merge_vector_items(items)

    def _merge_vector_items(self, items: list, x_gap: float = 180.0, y_thresh: float = 15.0) -> list:
        """
        Merges horizontally adjacent text items. PyMuPDF 'words' occasionally separates
        content living in distinct boxes (like GD&T cells [0.3] [S] [U]), even when
        they're far apart horizontally within the same row.

        Iterative: repeats single-pass merging until the list stabilises.
        (Previously recursive — risked RecursionError on complex drawings.)
        """
        while True:
            merged = self._merge_pass(items, x_gap, y_thresh)
            if len(merged) == len(items):
                return merged
            items = merged  # loop with a shorter list

    def _merge_pass(self, items: list, x_gap: float, y_thresh: float) -> list:
        """Single greedy merge pass — called repeatedly by _merge_vector_items."""
        if not items:
            return []

        merged = []
        skip = set()
        
        # Sort top-to-bottom then left-to-right
        sorted_items = sorted(items, key=lambda i: (
            sum(p[1] for p in i['bbox']) / 4,
            min(p[0] for p in i['bbox'])
        ))
        
        for i, item in enumerate(sorted_items):
            if i in skip:
                continue
                
            group = [item]
            skip.add(i)
            
            x1_max = max(p[0] for p in item['bbox'])
            cy1 = sum(p[1] for p in item['bbox']) / 4
            h1 = max(p[1] for p in item['bbox']) - min(p[1] for p in item['bbox'])
            
            for j in range(i + 1, len(sorted_items)):
                if j in skip:
                    continue
                    
                other = sorted_items[j]
                x2_min = min(p[0] for p in other['bbox'])
                x2_max = max(p[0] for p in other['bbox'])
                cy2 = sum(p[1] for p in other['bbox']) / 4
                h2 = max(p[1] for p in other['bbox']) - min(p[1] for p in other['bbox'])
                
                # Check horizontal alignment (center-y distance) and gap
                if abs(cy1 - cy2) < y_thresh:
                    gap = x2_min - x1_max
                    # Allow up to x_gap pixels of whitespace between items on the same row
                    if -15 <= gap <= x_gap:
                        group.append(other)
                        skip.add(j)
                        x1_max = max(x1_max, x2_max) # Extend right edge
                        cy1 = sum([sum(p[1] for p in g['bbox']) / 4 for g in group]) / len(group)
                        
            if len(group) > 1:
                group.sort(key=lambda g: min(p[0] for p in g['bbox']))
                merged_text = " ".join(g['text'] for g in group)
                
                xs = [p[0] for g in group for p in g['bbox']]
                ys = [p[1] for g in group for p in g['bbox']]
                
                new_bbox = [
                    [min(xs), min(ys)], [max(xs), min(ys)],
                    [max(xs), max(ys)], [min(xs), max(ys)]
                ]
                
                merged.append({
                    'bbox': new_bbox,
                    'text': merged_text,
                    'confidence': 1.0,
                    'source': 'vector',
                    'type': None
                })
            else:

                merged.append(item)

        return merged

    def filter_items(self, items: list, extractor) -> list:
        """
        Run token repair and classification on vector-extracted items.
        Reuses TextExtractor helpers so the same engineering-pattern rules apply.

        Args:
            items     : list of item dicts from extract_page_items()
            extractor : TextExtractor instance (already initialised in main.py)

        Returns:
            Filtered + classified list of item dicts.
        """
        result = []
        for item in items:
            text = item['text']

            # 1. Normalize spacing/symbol errors FIRST — vector PDFs often have
            #    decimals stored as "3 . 0" (spaces around the period) which would
            #    cause repair_merged_token to drop the token as noise.
            text = extractor.clean_text_content(text)
            if not text.strip():
                continue

            # 2. Repair split decimals (e.g. "3 . 0" already handled by clean,
            #    but this catches fused tokens like "2.720.5")
            text = extractor.repair_numeric_strings(text)

            # 3. Noise gate — drop tokens that don't match any engineering pattern.
            #    Now runs AFTER normalization so "3.0 D R F" passes correctly.
            repaired = extractor.repair_merged_token(text)
            if repaired is None:
                continue  # pure noise — drop

            item['text'] = repaired

            # 4. Classify the token type (DIMENSION, GDT, TEXT, etc.)
            item['type'] = extractor.classify_token(repaired)

            result.append(item)

        return result

