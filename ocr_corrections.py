"""
ocr_corrections.py — Per-drawing-series OCR correction maps.

Keys are exact OCR outputs. Values are the correct engineering text.

Rules:
  - GLOBAL_SYMBOL_CORRECTIONS   : safe on every drawing (systematic OCR font confusions)
  - DRAWING_SPECIFIC_CORRECTIONS: only correct for a known drawing/scanner batch.
    Only add entries when you've confirmed the error is systematic
    (same drawing series, same font, same scanner).
    Enable them by passing apply_drawing_specific=True in clean_text_content()
"""

# ── Systematic OCR errors regardless of drawing ───────────────────────────────
# Safe to apply globally: these are predictable glyph confusions on any scan.
GLOBAL_SYMBOL_CORRECTIONS = {
    'YIz':       'Y',
    'S00':       '0.05',
    'XYIZ':      'X Y Z',
    'XlYIZ':     'X Y Z',
    'Xi':        'X',
    # Truncated / garbled text tokens (OCR cuts last char at bounding box edge)
    'BREAK EDG': 'BREAK EDGE',
    'BREAK EDG.': 'BREAK EDGE',
    '&QR CORE':  '&QR CODE',
}

# ── Drawing-specific corrections (LENOVO scan batch / TBD drawing series) ─────
# Disabled by default.  Enable with apply_drawing_specific=True.
DRAWING_SPECIFIC_CORRECTIONS = {
    '313':   '3.4',
    '8 8':   '8.8',
    '8 8 6': '8.8',
}

