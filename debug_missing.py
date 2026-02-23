"""
Test PaddleOCR with lower DB detection thresholds to recover missed text regions.
"""
import os, sys
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import socket
_orig = socket.getaddrinfo
def _ipv4(h, p, f=0, t=0, pr=0, fl=0):
    return [r for r in _orig(h, p, socket.AF_INET, t, pr, fl) or []]
socket.getaddrinfo = _ipv4

import cv2, numpy as np
from paddleocr import PaddleOCR

TARGETS = ["364", "26.5", "26,5"]
TEMP_PNG = r"d:\new assing\_debug_page.png"

bgr = cv2.imread(TEMP_PNG)
image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
print(f"Image: {image.shape}")

OUT = r"d:\new assing\clahe_result.txt"

configs = [
    ("DEFAULT    det_thresh=0.3 box_thresh=0.6", dict(det_db_thresh=0.3, det_db_box_thresh=0.6)),
    ("SENSITIVE  det_thresh=0.2 box_thresh=0.4", dict(det_db_thresh=0.2, det_db_box_thresh=0.4)),
    ("VERY_SENS  det_thresh=0.1 box_thresh=0.3", dict(det_db_thresh=0.1, det_db_box_thresh=0.3)),
]

with open(OUT, "w") as f:
    for label, params in configs:
        ocr = PaddleOCR(use_angle_cls=True, lang='en',
                        use_gpu=False, show_log=False,
                        **params)
        raw = ocr.ocr(image)
        total = len(raw[0]) if raw and raw[0] else 0
        hits = []
        if raw and raw[0]:
            for line in raw[0]:
                bbox, (text, prob) = line
                if any(k in text for k in TARGETS):
                    cy = int(sum(p[1] for p in bbox) / 4)
                    hits.append(f"'{text}' conf={prob:.2f} cy={cy}")
        f.write(f"[{label}] total={total}\n")
        for h in hits:
            f.write(f"  HIT: {h}\n")
        if not hits:
            f.write(f"  NOT FOUND\n")
        f.flush()
        print(f"Done: {label} → {'FOUND: ' + str(hits) if hits else 'not found'}")

print("Written to clahe_result.txt")
