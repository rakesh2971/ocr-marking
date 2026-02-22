"""
Debug: Dump all raw OCR results in the bottom-right region of the page
(where ⊕ Ø0.5 A B C and F are located) to understand why they are missed.
"""
import cv2
import numpy as np
from processor import DocumentProcessor
from extractor import TextExtractor

def run():
    proc = DocumentProcessor()
    images = proc.pdf_to_images("LCA.pdf", zoom=2.0)
    img = images[0]
    h, w = img.shape[:2]

    extractor = TextExtractor()

    print(f"Image size: {w} x {h}")
    print("\n--- ALL raw EasyOCR results in bottom 35% of image ---")

    # Run raw OCR on the full image
    raw_results = extractor.reader.readtext(img, paragraph=False, contrast_ths=0.1, text_threshold=0.4)

    # Focus on the bottom 35% and right 50% where these annotations are
    y_cutoff = int(h * 0.60)
    x_cutoff = int(w * 0.40)

    count = 0
    for (bbox_raw, text, conf) in raw_results:
        xs = [p[0] for p in bbox_raw]
        ys = [p[1] for p in bbox_raw]
        cx = sum(xs) / 4
        cy = sum(ys) / 4
        if cy > y_cutoff and cx > x_cutoff:
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            print(f"  [{conf:.2f}] '{text}'  @ x={x_min}-{x_max}, y={y_min}-{y_max}")
            count += 1

    print(f"\nTotal items in that region: {count}")

    # Also draw ALL raw results in that region on the image for visual inspection
    debug_img = img.copy()
    for (bbox_raw, text, conf) in raw_results:
        xs = [p[0] for p in bbox_raw]
        ys = [p[1] for p in bbox_raw]
        cx = sum(xs) / 4
        cy = sum(ys) / 4
        if cy > y_cutoff and cx > x_cutoff:
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            color = (0, 255, 0) if conf > 0.5 else (0, 140, 255)
            cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(debug_img, f"{text[:15]} {conf:.2f}", (x_min, max(0, y_min - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Save scaled-down debug image
    out = cv2.resize(debug_img, (w // 3, h // 3))
    cv2.imwrite("debug_fp4_ocr.png", out)
    print("Saved debug_fp4_ocr.png")

if __name__ == "__main__":
    run()
