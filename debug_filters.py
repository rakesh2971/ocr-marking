"""
debug_filters.py — prints filter boundary diagnostics without running the full pipeline.
Usage: python debug_filters.py --input your_drawing.pdf
"""
import argparse
import os
import cv2
import numpy as np

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

from processor import DocumentProcessor
from filter import AnnotationFilter
from vector_extractor import VectorExtractor, is_vector_page
from extractor import TextExtractor

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--zoom", type=float, default=2.0)
parser.add_argument("--page", type=int, default=0, help="Page index (0-based)")
args = parser.parse_args()

print(f"\n=== DEBUG FILTER BOUNDARIES: {args.input} ===\n")

proc = DocumentProcessor()
fitz_doc = proc.open_doc(args.input)
page = fitz_doc[args.page]

# Get image
_, image = next(proc.iter_pages(fitz_doc, zoom=args.zoom))
if len(image.shape) == 3:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    image_gray = image

h, w = image_gray.shape
print(f"Image size: {w} x {h} px  (zoom={args.zoom})\n")

# Extract items
if is_vector_page(page):
    print("[vector page detected]")
    extractor = TextExtractor()
    vec = VectorExtractor()
    items = vec.extract_page_items(page, zoom=args.zoom)
    items = vec.filter_items(items, extractor)
else:
    print("[raster page detected]")
    extractor = TextExtractor()
    items = extractor.extract_text_custom(image, image_gray, x_threshold=40)

print(f"Total raw items extracted: {len(items)}\n")

# Run filters and print boundaries
annot_filter = AnnotationFilter()

# Notes section
print("--- NOTES FILTER ---")
valid, excl_notes = annot_filter.filter_notes_section(items)
print(f"  notes_cutoff_x = {annot_filter.notes_cutoff_x}")
print(f"  notes_cutoff_y = {annot_filter.notes_cutoff_y}")
print(f"  Items excluded: {len(excl_notes)}")
print(f"  Items kept:     {len(valid)}")
if excl_notes:
    print(f"  First 5 excluded: {[i['text'][:30] for i in excl_notes[:5]]}")
print()

# Table filter
print("--- TABLE FILTER ---")
valid2, excl_table = annot_filter.filter_bottom_right_table(valid, image.shape)
print(f"  _table_cutoff_y = {getattr(annot_filter, '_table_cutoff_y', 'NOT SET')}")
print(f"  Items excluded: {len(excl_table)}")
print(f"  Items kept:     {len(valid2)}")
print()

# Title block
print("--- TITLE BLOCK ---")
tb_x, tb_y = annot_filter.detect_title_block_boundary(valid2, image.shape)
print(f"  title_block_cutoff_x = {tb_x}")
print(f"  title_block_cutoff_y = {tb_y}")
print()

# Summary
print("--- SUMMARY ---")
print(f"  Page size:        {w} x {h}")
print(f"  Notes column:     x >= {annot_filter.notes_cutoff_x}  ({(annot_filter.notes_cutoff_x or 0)/w*100:.1f}% from left)")
print(f"  Notes top:        y >= {annot_filter.notes_cutoff_y}  ({(annot_filter.notes_cutoff_y or 0)/h*100:.1f}% from top)")
print(f"  Title block left: x >= {tb_x}  ({(tb_x or 0)/w*100:.1f}% from left)")
print(f"  Title block top:  y >= {tb_y}  ({(tb_y or 0)/h*100:.1f}% from top)")
print(f"  Table cutoff y:   {getattr(annot_filter, '_table_cutoff_y', 'NOT SET')}")
