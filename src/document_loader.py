"""
Document loader for CTI reports (PDF, HTML → clean text).
"""

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict, Any

def load_report(report_path: Path) -> Dict[str, Any]:
    """
    Load CTI report and extract clean text.
    
    Args:
        report_path: Path to PDF or HTML file
        
    Returns:
        dict with 'doc_id', 'title', 'text'
    """
    doc_id = report_path.stem
    
    if report_path.suffix.lower() == '.pdf':
        return _load_pdf(report_path, doc_id)
    elif report_path.suffix.lower() in ['.html', '.htm']:
        return _load_html(report_path, doc_id)
    else:
        raise ValueError(f"Unsupported format: {report_path.suffix}")

def _load_pdf(pdf_path: Path, doc_id: str) -> Dict[str, Any]:
    """Extract text from PDF."""
    doc = fitz.open(pdf_path)
    
    text_parts = []
    title = None
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Try to extract title from first page header
        if page_num == 0 and title is None:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks[:3]:  # First few blocks
                if "lines" in block:
                    first_line = block["lines"][0]["spans"][0]["text"].strip()
                    if first_line and len(first_line) > 5:
                        title = first_line
                        break
        
        # Extract page text
        text_parts.append(page.get_text("text"))
    
    doc.close()
    
    full_text = "\n\n".join(text_parts)
    title = title or doc_id
    
    return {
        "doc_id": doc_id,
        "title": title,
        "text": full_text.strip()
    }

def _load_html(html_path: Path, doc_id: str) -> Dict[str, Any]:
    """Extract text from HTML."""
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
    
    # Extract title
    title = soup.title.string.strip() if soup.title else doc_id
    
    # Extract main text (prioritise article, main, body)
    text_selectors = ['article', 'main', 'body', '[role="main"]']
    text_elem = None
    for selector in text_selectors:
        text_elem = soup.select_one(selector)
        if text_elem:
            break
    
    if text_elem:
        full_text = text_elem.get_text(separator='\n', strip=True)
    else:
        full_text = soup.get_text(separator='\n', strip=True)
    
    return {
        "doc_id": doc_id,
        "title": title,
        "text": full_text
    }
