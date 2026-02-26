import argparse
import os
import csv
import re
import cv2
import numpy as np
import fitz

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
from vector_extractor import VectorExtractor, is_vector_page


def _dedup_vertical_items(vert_items, valid_items, image_shape):
    """
    For each vertical item, decide if it duplicates an existing horizontal item.
    Returns (deduped_vert, items_to_remove_from_horizontal).

    Fix: the previous inline loop used `continue` after appending to
    horizontal_to_remove, which meant `is_dup` was never set for tall boxes,
    so the vertical item was always added even when it spatially overlapped.
    """
    h_img, w_img = image_shape[:2]
    deduped_vert = []
    horizontal_to_remove = []

    from box_detector import _build_spatial_index, _get_nearby
    spatial_index = _build_spatial_index(valid_items, cell_size=200)

    for v_item in vert_items:
        v_bbox = v_item['bbox']
        v_x_min = min(p[0] for p in v_bbox)
        v_x_max = max(p[0] for p in v_bbox)
        v_y_min = min(p[1] for p in v_bbox)
        v_y_max = max(p[1] for p in v_bbox)

        # Skip items in the notes/title block (right-hand side)
        v_cx = (v_x_min + v_x_max) / 2
        v_cy = (v_y_min + v_y_max) / 2
        if (v_cx > w_img * 0.70 and v_cy > h_img * 0.80) or \
           (v_cx > w_img * 0.72 and v_cy < h_img * 0.85):
            continue

        overlapping_h_item = None
        nearby_h_items = _get_nearby(spatial_index, v_cx, v_cy, cell_size=200)
        
        for h_item in nearby_h_items:
            h_bbox = h_item['bbox']
            h_x_min = min(p[0] for p in h_bbox)
            h_x_max = max(p[0] for p in h_bbox)
            h_y_min = min(p[1] for p in h_bbox)
            h_y_max = max(p[1] for p in h_bbox)

            no_overlap = (v_x_max < h_x_min - 5 or v_x_min > h_x_max + 5 or
                          v_y_max < h_y_min - 5 or v_y_min > h_y_max + 5)
            if no_overlap:
                continue

            overlapping_h_item = h_item
            break


        if overlapping_h_item is None:
            # No overlap — vertical item is genuinely new
            deduped_vert.append(v_item)
        else:
            h_bbox = overlapping_h_item['bbox']
            h_w = max(p[0] for p in h_bbox) - min(p[0] for p in h_bbox)
            h_h = max(p[1] for p in h_bbox) - min(p[1] for p in h_bbox)
            h_text = overlapping_h_item['text'].strip()
            alnum_count = sum(c.isalnum() for c in h_text)

            # Prefer vertical item if the horizontal item is:
            # - tall relative to its width (likely a mis-read vertical line), OR
            # - very short alphanumeric (likely a fragment / noise)
            h_is_weak = (h_h > h_w * 1.2) or (alnum_count < 3 and len(h_text) <= 4)

            if h_is_weak:
                horizontal_to_remove.append(overlapping_h_item)
                deduped_vert.append(v_item)
            # else: horizontal item is strong — keep it, discard vertical

    return deduped_vert, horizontal_to_remove


def _apply_horizontal_removals(valid_items, horizontal_to_remove):
    """Remove items by identity (not value equality) to avoid dropping wrong items
    when multiple annotations share the same text and coordinates."""
    remove_ids = {id(it) for it in horizontal_to_remove}
    return [it for it in valid_items if id(it) not in remove_ids]


def main():
    parser = argparse.ArgumentParser(description="Automated Drawing Marking System")
    parser.add_argument("--input",      required=True, help="Input PDF file path")
    parser.add_argument("--output_pdf", required=True, help="Output Annotated PDF file path")
    parser.add_argument("--output_csv", required=True, help="Output CSV file path")
    parser.add_argument("--zoom", type=float, default=2.0, help="Zoom factor (default: 2.0)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    print(f"Processing {args.input}...")

    from pipeline import AnnotationPipeline
    pipeline = AnnotationPipeline(args)
    
    # ── Save annotated PDF ──────────────────────────────────────────────────
    print(f"Opening output stream for {args.output_pdf}...")
    output_doc = fitz.open()
    all_mappings = []
    
    # Iterate over the generator, incrementally buffering and dropping RAM
    for page_idx, ann_img, mappings in pipeline.run():
        pipeline.processor.append_page_to_pdf(output_doc, ann_img)
        all_mappings.extend(mappings)
        del ann_img # explicit free
        
    import tempfile
    import shutil
    
    out_dir = os.path.dirname(os.path.abspath(args.output_pdf))
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pdf', dir=out_dir)
    os.close(tmp_fd)

    try:
        output_doc.save(tmp_path)
        output_doc.close()
        shutil.move(tmp_path, args.output_pdf)
        print(f"Saved annotated PDF to {args.output_pdf}")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed to save PDF to '{args.output_pdf}' (Disk full?): {e}") from e

    # ── Save CSV ────────────────────────────────────────────────────────────
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
