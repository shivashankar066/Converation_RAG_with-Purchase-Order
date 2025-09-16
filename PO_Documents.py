import re
import json
import pandas as pd
from pathlib import Path
import pdfplumber
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage
from datetime import datetime

# -------------------------
# Config
# -------------------------
LLM_MODEL = "llama3.1:8b"
TEMPERATURE = 0.0

# Initialize Ollama LLM
llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)

# ---------- helpers ----------
MON_MAP = {'JAN':'01','FEB':'02','MAR':'03','APR':'04','MAY':'05','JUN':'06',
           'JUL':'07','AUG':'08','SEP':'09','SEPT':'09','OCT':'10','NOV':'11','DEC':'12'}

def normalize_date(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    t = text.strip().replace('.', '').replace('/', ' ').upper()
    m = re.search(r'(\d{1,2})[-\s]+([A-Z]{3,9})[-\s]+(\d{2,4})', t)
    if m:
        dd, mon, yy = m.groups()
        mm = MON_MAP.get(mon[:3].upper(), '01')
        yy = int(yy)
        if yy < 100: yy = 2000 + yy
        try:
            return datetime(yy, int(mm), int(dd)).strftime("%Y-%m-%d")
        except:
            return ""
    m2 = re.match(r'(\d{4}-\d{2}-\d{2})', text.strip())
    if m2:
        return m2.group(1)
    return ""

def safe_json_parse(text: str):
    if not text or not isinstance(text, str):
        return {}
    try:
        return json.loads(text)
    except:
        m = re.search(r'\{[\s\S]*\}', str(text))
        if m:
            cand = m.group(0)
            cand2 = cand.replace("’", "'").replace("“", '"').replace("”", '"')
            cand2 = re.sub(r'([\{,]\s*)([A-Za-z0-9_]+)\s*:', r'\1"\2":', cand2)
            cand2 = cand2.replace("'", '"')
            cand2 = re.sub(r',\s*([\}\]])', r'\1', cand2)
            try:
                return json.loads(cand2)
            except:
                return {}
    return {}

# ---------- PDF text reading ----------
def read_pdf_pages(file_obj):
    pages = []
    with pdfplumber.open(file_obj) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return pages

# ---------- deterministic block splitting ----------
def find_part_blocks(combined_text: str):
    pattern = re.compile(
        r'QUANTITY\s*=\s*([0-9,]+\.\d+|\d+)\s*([A-Z]{1,5})\s+PART\s*=\s*([A-Z0-9\-\._\/]+)',
        re.IGNORECASE
    )
    starts = list(pattern.finditer(combined_text))
    blocks = []
    for i, m in enumerate(starts):
        sidx = m.start()
        next_start = starts[i+1].start() if i+1 < len(starts) else None
        eidx = next_start if next_start else len(combined_text)
        block_text = combined_text[sidx:eidx]
        blocks.append({
            "part_no": m.group(3).strip(),
            "qty_raw": m.group(1).replace(',', '').strip(),
            "unit_hint": m.group(2).strip(),
            "start": sidx,
            "end": eidx,
            "text": block_text
        })
    return blocks

# ---------- fallback extractors ----------
def fallback_extract_item(combined_text, start, end):
    left_window = combined_text[max(0, start - 300): start]
    left_matches = list(re.finditer(r'(^|\n)\s*([0-9]{3,4})\b', left_window))
    if left_matches:
        return left_matches[-1].group(2)
    right_window = combined_text[end: end + 200]
    m = re.search(r'\b([0-9]{3,4})\b', right_window)
    if m:
        return m.group(1)
    return ""

def fallback_extract_part_description(block_text, part_no):
    """
    Extracts the part description lines, skipping separators and schedule lines.
    """
    lines = [ln.strip() for ln in block_text.splitlines() if ln.strip()]
    desc_parts = []
    idx = None
    for i, ln in enumerate(lines):
        if re.search(r'PART\s*=\s*' + re.escape(part_no), ln, re.IGNORECASE):
            idx = i
            break
    if idx is None:
        return ""

    for ln in lines[idx+1: idx+8]:
        # stop if control keywords appear
        if re.search(r'(EPR ISSUE|DRAWING ISSUE|DATE|ITEM AMENDMENT|INSPECTION)', ln, re.IGNORECASE):
            break
        # skip unwanted lines
        if re.match(r'[-]{3,}', ln):
            continue
        if "RESCHEDULE RIGHT (OUT)" in ln.upper():
            continue
        if "COMMITTED DELIVERY" in ln.upper():
            continue
        desc_parts.append(ln)

    return " ".join(desc_parts).strip()


def fallback_extract_unit_price(block_text):
    m = re.search(r'\$\s*([0-9,]+\.\d{1,4})', block_text)
    if m: return m.group(1)
    m2 = re.search(r'([0-9,]+\.\d{1,4})\s*EA', block_text, re.IGNORECASE)
    if m2: return m2.group(1)
    return ""

def fallback_extract_epr_and_drawing(block_text):
    m = re.search(r"EPR\s*ISSUE\s*([A-Z0-9\-]+)\s+DRAWING\s*ISSUE\s*([A-Z0-9\-]+)", block_text, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", ""

def fallback_extract_delivery_date(combined_text, start, end, block_text, po_date):
    patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{1,2}[-\s][A-Z]{3}[-\s]\d{2,4}',
        r'\d{1,2}\s+[A-Z]{3,9}\s+\d{4}'
    ]
    for pat in patterns:
        m = re.search(pat, block_text)
        if m:
            norm = normalize_date(m.group(0))
            if norm and norm != po_date:
                return norm
    return ""

# ---------- LLM calls ----------
def extract_part_with_llm(block_text: str, po_date: str):
    parsed = {
        "Item": "", "Part_No": "", "Part_Description": "", "Qty": "",
        "Unit": "", "Unit_Price": "", "EPR_Issue": "", "Drawing_Issue": "",
        "Delivery_Date": ""
    }
    # Part No
    m_part = re.search(r'PART\s*=\s*([A-Z0-9\-\._\/]+)', block_text)
    if m_part: parsed["Part_No"] = m_part.group(1).strip()
    # Qty + Unit
    m_qty = re.search(r'QUANTITY\s*=\s*([0-9,]+\.\d+|\d+)\s*([A-Z]{1,5})', block_text)
    if m_qty:
        parsed["Qty"] = m_qty.group(1).replace(',', '')
        parsed["Unit"] = m_qty.group(2)
    # Price
    parsed["Unit_Price"] = fallback_extract_unit_price(block_text)
    # EPR + Drawing
    epr, drawing = fallback_extract_epr_and_drawing(block_text)
    parsed["EPR_Issue"], parsed["Drawing_Issue"] = epr, drawing
    # Delivery Date
    m_date = re.search(r'DATE\s+([0-9]{1,2}\s+[A-Z]{3,9}\s+\d{2,4})', block_text)
    if m_date: parsed["Delivery_Date"] = normalize_date(m_date.group(1))
    return json.dumps(parsed)

# ---------- header extract ----------
def extract_header_info(first_page_text: str):
    po_no, po_date = "", ""
    m_no = re.search(r'Purchase Order No\.?\s*([A-Z0-9\-]+)', first_page_text, re.IGNORECASE)
    if m_no:
        po_no = m_no.group(1).strip()
    m_date = re.search(r'Date\s+([0-9]{1,2}[-\s][A-Z]{3}[-\s]\d{2,4})', first_page_text, re.IGNORECASE)
    if m_date:
        po_date = normalize_date(m_date.group(1).strip())
    return po_no, po_date

# ---------- run for one PDF ----------
def run_extraction(file_obj, file_name: str):
    pages = read_pdf_pages(file_obj)
    combined = "\n".join([pages[i] for i in range(len(pages)) if pages[i]])
    po_no, po_date = extract_header_info(pages[0]) if pages else ("", "")
    blocks = find_part_blocks(combined)
    rows = []
    for b in blocks:
        raw = extract_part_with_llm(b["text"], po_date)
        parsed = safe_json_parse(raw)
        if not parsed.get("Item"):
            parsed["Item"] = fallback_extract_item(combined, b["start"], b["end"]) or ""
        if not parsed.get("Part_No"):
            parsed["Part_No"] = b["part_no"]
        if not parsed.get("Part_Description") and parsed.get("Part_No"):
            parsed["Part_Description"] = fallback_extract_part_description(b["text"], parsed["Part_No"])
        if not parsed.get("Delivery_Date"):
            parsed["Delivery_Date"] = fallback_extract_delivery_date(combined, b["start"], b["end"], b["text"], po_date)
        # pad item
        it = parsed.get("Item", "")
        if re.match(r'^\d+$', str(it)):
            parsed["Item"] = str(it).zfill(4)
        parsed["PO_No"] = po_no
        parsed["PO_Date"] = po_date
        parsed["Source_File"] = file_name
        rows.append(parsed)
    return rows

# ---------- Streamlit UI ----------
st.title("📄 Purchase Order PDF Extractor")

uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_rows = []
    for file in uploaded_files:
        rows = run_extraction(file, file.name)
        all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows, columns=[
            "Item","Part_No","Part_Description","Qty","Unit","Unit_Price",
            "EPR_Issue","Drawing_Issue","Delivery_Date","PO_No","PO_Date","Source_File"
        ])
        st.dataframe(df)
        # Export Excel
        out_file = "extracted_po.xlsx"
        df.to_excel(out_file, index=False)
        with open(out_file, "rb") as f:
            st.download_button("📥 Download Excel", f, file_name=out_file)
