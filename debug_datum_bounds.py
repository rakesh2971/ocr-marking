from processor import DocumentProcessor
from filter import AnnotationFilter
from vector_extractor import VectorExtractor

def check_adjacent_datums():
    p = DocumentProcessor()
    fitz_doc = p.open_doc('LCA_1.pdf')
    vec = VectorExtractor()
    items = vec.extract_page_items(fitz_doc[0], zoom=2.0)
    
    af = AnnotationFilter()
    img = p.pdf_to_images('LCA_1.pdf', zoom=2.0)[0]
    circles = af.detect_black_circles(img)
    print(f"Detected {len(circles)} datum circles")

    for (cx, cy, cr) in circles:
        # Check for text near this circle
        nearby = []
        for it in items:
            bbox = it['bbox']
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            tcx = sum(x_coords) / 4
            tcy = sum(y_coords) / 4
            
            # Check bounding box distance
            dist = ((tcx - cx)**2 + (tcy - cy)**2)**0.5
            
            # If it's inside or very close outside (like attached coordinate)
            if dist < cr * 3:
                nearby.append((dist, it['text'].strip(), tcx, tcy))
        
        if len(nearby) > 0:
            texts = [t[1] for t in nearby]
            # Has a datum identifier?
            if any(t in ['X', 'Y', 'Z'] for t in texts):
                print(f"Datum at ({cx:.1f}, {cy:.1f}) r={cr:.1f}:")
                for dist, text, tcx, tcy in sorted(nearby):
                    print(f"  {text:5s} at ({tcx:.1f}, {tcy:.1f}), dist={dist:.1f}, is_inside={dist < cr*1.3}")

if __name__ == "__main__":
    check_adjacent_datums()
