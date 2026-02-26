"""
Lightweight debug: trace right-side items through each filter stage.
Skips PaddleOCR init — only uses the vector extractor + filter.
"""
import sys, os
sys.path.insert(0, r"d:\new assing")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import fitz
import cv2
import numpy as np

PDF_PATH  = r"d:\new assing\LCA_1.pdf"
ZOOM      = 2.0
X_SPLIT   = 0.55   # "right side" = center-x > 55% of image width

doc  = fitz.open(PDF_PATH)
page = doc[0]
mat  = fitz.Matrix(ZOOM, ZOOM)
pix  = page.get_pixmap(matrix=mat)
buf  = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_h, img_w = image.shape[:2]
right_x_cut = img_w * X_SPLIT
print(f"Image: {img_w}x{img_h}  |  right-side threshold x>{right_x_cut:.0f}")

# ── Use vector extractor (no PaddleOCR) ────────────────────────────────────
from vector_extractor import VectorExtractor

class _FakeExtractor:
    """Minimal stand-in — skips repair/classify so we don't need PaddleOCR."""
    def repair_merged_token(self, t):   return t
    def clean_text_content(self, t):    
        import re
        t = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', t)
        return t.strip()
    def repair_numeric_strings(self, t): return t
    def classify_token(self, t):         return "OTHER"
    def merge_gdt_rows(self, items, **kw): return items

vec_ext = VectorExtractor()
fake_ex = _FakeExtractor()

raw_items    = vec_ext.extract_page_items(page, zoom=ZOOM)
stage1_items = vec_ext.filter_items(raw_items, fake_ex)
# Skip merge_gdt_rows for simplicity

def count_right(items, label):
    right = [it for it in items if sum(p[0] for p in it['bbox'])/4 > right_x_cut]
    print(f"  [{label}] total={len(items)}  right={len(right)}")
    return right

r = count_right(raw_items,    "raw vector words")
r = count_right(stage1_items, "after fake filter_items")

# ── Filter stages ──────────────────────────────────────────────────────────
from filter import AnnotationFilter
ann = AnnotationFilter()

print("\n--- Circle detection ---")
circles = ann.detect_black_circles(image_gray)
print(f"  {len(circles)} circles found")

after_circ, excl_circ = ann.filter_by_circles(stage1_items, circles)
rc = count_right(after_circ, "after filter_by_circles")
right_excl_circ = [it for it in excl_circ if sum(p[0] for p in it['bbox'])/4 > right_x_cut]
print(f"  Right-side items REMOVED by circle filter: {len(right_excl_circ)}")
for it in right_excl_circ[:8]:
    print(f"    x={sum(p[0] for p in it['bbox'])/4:.0f}  text={it['text']!r}")

print("\n--- Notes filter ---")
after_notes, excl_notes = ann.filter_notes_section(after_circ)
rn = count_right(after_notes, "after filter_notes_section")
ren = [it for it in excl_notes if sum(p[0] for p in it['bbox'])/4 > right_x_cut]
print(f"  Right-side items REMOVED by notes filter: {len(ren)}")
for it in ren[:8]:
    print(f"    x={sum(p[0] for p in it['bbox'])/4:.0f}  text={it['text']!r}")

print("\n--- View label filter ---")
after_view, excl_view = ann.filter_view_labels(after_notes)
rv = count_right(after_view, "after filter_view_labels")
rev = [it for it in excl_view if sum(p[0] for p in it['bbox'])/4 > right_x_cut]
print(f"  Right-side items REMOVED by view filter: {len(rev)}")
for it in rev[:8]:
    print(f"    x={sum(p[0] for p in it['bbox'])/4:.0f}  text={it['text']!r}")

print("\n--- Table filter ---")
after_table, excl_table = ann.filter_bottom_right_table(after_view, image.shape)
rt = count_right(after_table, "after filter_bottom_right_table")

print("\n=== FINAL right-side items ===", len(rt))
for it in rt[:15]:
    print(f"  x={sum(p[0] for p in it['bbox'])/4:.0f}  text={it['text']!r}")
