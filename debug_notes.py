from filter import AnnotationFilter
from vector_extractor import VectorExtractor
from processor import DocumentProcessor

def run_debug():
    p = DocumentProcessor()
    fitz_doc = p.open_doc('LCA_1.pdf')
    vec = VectorExtractor()
    items = vec.extract_page_items(fitz_doc[0], zoom=2.0)
    af = AnnotationFilter()
    
    notes_header = None
    for i in items:
        if i['text'].strip().upper() in ['NOTES', 'NOTE']:
            notes_header = i
            break
            
    if notes_header:
        print('Notes header found at', notes_header['bbox'], 'text:', notes_header['text'])
        bbox = notes_header['bbox']
        header_x_min = min(p[0] for p in bbox)
        cutoff_x = header_x_min - 20
        print('Cutoff X calculation:', cutoff_x)
        
        excluded = []
        for it in items:
            item_bbox = it['bbox']
            item_x_min = min(p[0] for p in item_bbox)
            if item_x_min >= cutoff_x:
                excluded.append(it['text'])
                
        print(f"Items excluded: {len(excluded)}")
        print("Sample of excluded items on the right side:")
        print(excluded[:20])
    else:
        print('No notes header found')

if __name__ == "__main__":
    run_debug()
