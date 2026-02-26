# config.py — single source of truth for all tunable constants
#
# All pixel-level thresholds are calibrated at BASELINE_ZOOM = 2.0.
# At runtime they are scaled by (actual_zoom / BASELINE_ZOOM) ** power
# where power=2 for areas and power=1 for lengths/radii/distances.

# ── Zoom baseline ────────────────────────────────────────────────────────────
BASELINE_ZOOM = 2.0

# ── Box detector (box_detector.py) ───────────────────────────────────────────
# Datum boxes at zoom=2.0 are roughly 20×20px to 67×67px squares.
BOX_AREA_MIN        = 800     # px² at baseline zoom
BOX_AREA_MAX        = 4500    # px² at baseline zoom
BOX_ASPECT_MIN      = 0.6     # width/height ratio minimum
BOX_ASPECT_MAX      = 2.8     # width/height ratio maximum
BOX_FILL_RATIO      = 0.85    # contour area / bounding-box area
BOX_EDGE_MARGIN     = 50      # px — ignore boxes within this margin of image edge
BOX_ISOLATION_MARGIN = 5      # px — gap below which two boxes are "touching"
BOX_OCR_UPSCALE     = 3.0    # upscale factor when OCR-ing individual box crops
BOX_OCR_CONF_MIN    = 0.10   # minimum OCR confidence to accept a boxed character
BOX_MAX_CHARS       = 4      # maximum character length of a valid datum marker

# ── GD&T frame detector ──────────────────────────────────────────────────────
GDT_AREA_MIN        = 700
GDT_AREA_MAX        = 20000
GDT_ASPECT_MIN      = 0.4
GDT_ASPECT_MAX      = 10.0
GDT_FILL_RATIO      = 0.75
GDT_EDGE_MARGIN     = 50      # px — same edge guard as datum boxes
GDT_OCR_CONF_MIN    = 0.15
GDT_MAX_TEXT_LEN    = 25

# ── Duplicate removal (box_detector.py _remove_duplicates) ──────────────────
DEDUP_OVERLAP_THRESHOLD = 0.50   # fraction of box area that must overlap
DEDUP_CENTER_PROXIMITY  = 40     # px — centers closer than this = same item

# ── Clustering (clustering.py) ───────────────────────────────────────────────
CLUSTER_DILATION_KERNEL    = 90      # px ellipse kernel for morphological dilation
CLUSTER_MIN_AREA_FRACTION  = 0.0015  # fraction of page area for a valid cluster
CLUSTER_ROW_HEIGHT_DEFAULT = 600     # px — nominal row height for reading order
CLUSTER_ROW_HEIGHT_MIN     = 400     # px — minimum (avoids over-splitting tall pages)

# ── Annotation filter ────────────────────────────────────────────────────────
NOTES_CUTOFF_MARGIN   = 20    # px left of NOTES header to set exclusion cutoff
TABLE_SCAN_FRACTION   = 0.35  # fraction of page height to scan above TABLE NO
CIRCLE_EXPANDED_RATIO = 1.3   # radius multiplier for corner-check on vector circles

# ── Visualizer ───────────────────────────────────────────────────────────────
BALLOON_RADIUS          = 25
BALLOON_DISTANCES       = [50, 75, 100, 125, 150]   # px distances tried for placement
BALLOON_DARK_THRESHOLD  = 10   # dark pixels below this = clear spot for balloon
BALLOON_FONT_SCALE      = 1.0
BALLOON_THICKNESS       = 2


# ── Scaling helpers ──────────────────────────────────────────────────────────

def scale_area(base_area: float, zoom: float) -> float:
    """Scale a pixel-area threshold from BASELINE_ZOOM to actual zoom."""
    return base_area * (zoom / BASELINE_ZOOM) ** 2


def scale_length(base_length: float, zoom: float) -> float:
    """Scale a pixel-length/radius/distance threshold from BASELINE_ZOOM to actual zoom."""
    return base_length * (zoom / BASELINE_ZOOM)
