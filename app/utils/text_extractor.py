from datetime import datetime
from enum import Enum
import io 
import csv 
import time
from typing import Any, List, Optional, Dict
import asyncio
from pypdf import PdfReader
import pdfplumber
import docx
from  app.utils.logger import get_logger 

logger = get_logger(__name__)

def extract_pdf(file_bytes: bytes) -> Dict[str, Any]:
    pages_text:List[str] = []
    metadata: Dict[str, Any] = {
        "pages": [],
        'page_count': 0
    }
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf :
            metadata["page_count"] = len(pdf.pages)
            for idx, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ''
                pages_text.append(text)
                metadata["pages"].append({
                    'page_number': idx,
                    'char_count':len(text)
                })
                
    except Exception:
        logger.warning("pdfplumber failed failing back to PyPDF2 ")
        
        reader = PdfReader(io.BytesIO(file_bytes))
        metadata["page_count"] = len(reader.pages)
        for page in reader.pages:
            pages_text.append(page.extract_text() or '')
            
    full_text = "\n\n".join(pages_text)
    metadata['word_count'] = len(full_text)
    return {'text': full_text, "metadata": metadata}


def extract_txt(file_bytes: bytes) -> Dict[str, Any]:
    
    for en in ('utf-8', 'latin-1', 'cp1252'):
        try:
            text = file_bytes.decode(en)
            return {
                'text': text,
                "metadata": {
                    'encoding':en,
                    'word_count':len(text),
                    'page_count': 1
                }
            }
            
        except UnicodeDecodeError:
            continue
        
    raise ValueError("Cannot decode text file")

def extract_docx(file_bytes: bytes) -> Dict[str, Any]:
    doc = docx.Document(io.BytesIO(file_bytes))
    parts: List[str] = []
    for part in doc.paragraphs:
        if part.text.strip():
            parts.append(part.text)
            
    text = "\n".join(parts)
    return {
        "text": text,
        'metadata':{
            'word_count': len(text),
            'page_count': 1
        }
    } 
    
    
def extract_csv(file_bytes: bytes) -> Dict[str, Any]:
    
    decoded = file_bytes.decode('utf-8')
    rows = csv.reader(io.StringIO(decoded))
    rows = ['|' .join(row) for row in rows]
    text = "\n".join(rows)
    return {
        "text": text,
        'metadata': {
            'word_count': len(text),
            'page_count': 1
        }
    }


_EXTRACTORS = {
    "pdf": extract_pdf,
    "txt": extract_txt,
    "docx": extract_docx,
    "md": extract_txt, 
    "csv": extract_csv,
}


async def extract_text(file_bytes: bytes, file_type: str):
    extractor = _EXTRACTORS.get(file_type)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {file_type}")
    return await asyncio.to_thread(extractor, file_bytes)
