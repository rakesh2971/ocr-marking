
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import io
import os

class DocumentProcessor:
    def __init__(self):
        pass

    def open_doc(self, pdf_path):
        """
        Return the raw fitz.Document for the given PDF.
        Used by the vector path so the document is opened once and reused
        across all pages without re-opening the file per page.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if os.path.getsize(pdf_path) == 0:
            raise ValueError(f"PDF file is empty: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
        except fitz.FileDataError as e:
            raise RuntimeError(
                f"Could not open '{pdf_path}'. "
                f"File may be corrupt, password-protected, or not a PDF.\n"
                f"Original error: {e}"
            ) from e
        if doc.is_encrypted:
            raise RuntimeError(
                f"'{pdf_path}' is password-protected. "
                f"Decrypt it first before processing."
            )
        if doc.page_count == 0:
            raise RuntimeError(f"'{pdf_path}' has no pages.")
        return doc

    def pdf_to_images(self, pdf_path, zoom=2.0):
        """
        Converts a PDF to a list of images (numpy arrays for OpenCV).
        WARNING: Buffers all pages in RAM. Use iter_pages for large documents.
        """
        return list(self.iter_pages(pdf_path, zoom))

    def iter_pages(self, pdf_path_or_doc, zoom=2.0):
        """
        Yields (page_index, image) one at a time instead of loading all into RAM.
        Essential for large engineering drawing sets.
        Accepts either a file path (str) or an already-open fitz.Document.
        """
        if isinstance(pdf_path_or_doc, str):
            doc = fitz.open(pdf_path_or_doc)
            close_after = True
        else:
            doc = pdf_path_or_doc
            close_after = False

        mat = fitz.Matrix(zoom, zoom)
        
        for i, page in enumerate(doc):
            try:
                pix = page.get_pixmap(matrix=mat)
                if pix.h == 0 or pix.w == 0:
                    print(f"Warning: Page {i + 1} rendered as empty image — skipping.")
                    continue
                    
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                
                if pix.n == 4:
                    img_data = img_data[:, :, :3]
                    
                yield i, img_data
                # Explicitly free memory (uncompressed high-res images are massive)
                del pix
                del img_data
            except Exception as e:
                print(f"Error rendering page {i + 1}: {e} — skipping.")
                continue

        if close_after:
            doc.close()

    def images_to_pdf(self, images, output_path):
        """
        Converts a list of images to a PDF.
        WARNING: Requires all images in RAM. Use append_page_to_pdf for large documents.
        """
        import tempfile
        import shutil
        
        out_dir = os.path.dirname(os.path.abspath(output_path))
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pdf', dir=out_dir)
        os.close(tmp_fd)
        
        try:
            doc = fitz.open()
            for img in images:
                self.append_page_to_pdf(doc, img)
            doc.save(tmp_path)
            doc.close()
            
            shutil.move(tmp_path, output_path)
            print(f"Saved annotated PDF to {output_path}")
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(f"Failed to save PDF to '{output_path}': {e}") from e

    def append_page_to_pdf(self, doc, img):
        """
        Appends a single image to an open fitz Document.
        """
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img)
        else:
            img_pil = img
            
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='JPEG', quality=95)
        
        page = doc.new_page(width=img_pil.width, height=img_pil.height)
        page.insert_image(page.rect, stream=img_byte_arr.getvalue())
