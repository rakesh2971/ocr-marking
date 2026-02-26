"""
debug_missing_dims.py
Traces where 18.99 and 26.53 disappear in the raster pipeline.
Prints raw PaddleOCR results → after repair → after cleanup → after merge.
"""
import sys, os
sys.path.insert(0, r"d:\new assing")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import fitz
import cv2, numpy as np

PDF_PATH = r"d:\new assing\LCA.pdf"
ZOOM = 2.0
TARGETS = {"18.99", "26.53", "18", "26", "18.9", "26.5", "1899", "2653"}

doc  = fitz.open(PDF_PATH)
page = doc[0]
mat  = fitz.Matrix(ZOOM, ZOOM)
pix  = page.get_pixmap(matrix=mat)
buf  = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

print("=" * 60)
print("STEP 1: Raw PaddleOCR output")
print("=" * 60)

from extractor import TextExtractor
ext = TextExtractor()

# Access the raw PaddleOCR result before any post-processing
raw_result = ext.ocr.ocr(image, cls=True)
raw_items = []
if raw_result and raw_result[0]:
    for line in raw_result[0]:
        bbox, (text, prob) = line
        raw_items.append({'bbox': bbox, 'text': text, 'confidence': prob})

# Sort by Y then X
raw_items.sort(key=lambda x: (sum(p[1] for p in x['bbox'])/4, sum(p[0] for p in x['bbox'])/4))

print(f"Total raw OCR tokens: {len(raw_items)}")
print("\n--- Tokens near targets (18.xx or 26.xx range) ---")
for it in raw_items:
    t = it['text']
    # Show any token that contains 18 or 26
    if any(tgt in t for tgt in ['18', '26', '1899', '2653']):
        cx = sum(p[0] for p in it['bbox'])/4
        cy = sum(p[1] for p in it['bbox'])/4
        print(f"  RAW: '{t}' conf={it['confidence']:.2f} @ ({cx:.0f}, {cy:.0f})")

print("\n--- ALL tokens sorted by position (Y then X) ---")
for it in raw_items:
    cx = sum(p[0] for p in it['bbox'])/4
    cy = sum(p[1] for p in it['bbox'])/4
    print(f"  y={cy:5.0f}  x={cx:5.0f}  conf={it['confidence']:.2f}  '{it['text']}'")

print("\n" + "=" * 60)
print("STEP 2: After repair_merged_token (what gets dropped)")
print("=" * 60)

for it in raw_items:
    t = it['text'].strip()
    # Apply repair_numeric_strings first
    t2 = ext.repair_numeric_strings(t)
    result = ext.repair_merged_token(t2)
    if result is None:
        cx = sum(p[0] for p in it['bbox'])/4
        cy = sum(p[1] for p in it['bbox'])/4
        if any(tgt in it['text'] for tgt in ['18', '26']):
            print(f"  DROPPED: '{it['text']}' → repair → None @ ({cx:.0f},{cy:.0f})")
