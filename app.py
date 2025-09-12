import streamlit as st
import re
from pathlib import Path
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ---------------- CONFIG ----------------
class Config:
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
    OLLAMA_CHAT_MODEL = "llama3.1:8b"

# --------- PDF Parsing ----------
PART_START = re.compile(
    r'QUANTITY\s*=\s*([0-9.,]+)\s*(\w+)?\s*PART\s*=\s*([A-Za-z0-9\-\./]+)',
    re.IGNORECASE
)

def read_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    full = "\n".join(f"[[PAGE {p}]]\n{t}" for p, t in pages)
    return full

def parse_header(text):
    header = {}

    # Capture Deliver To block (multi-line, keep intact)
    deliver = re.search(r'Deliver to:\s*(.*?)(?=Invoice to:|Conditions of purchase:)', text,
                        re.DOTALL | re.IGNORECASE)
    if deliver:
        lines = [l.strip() for l in deliver.group(1).splitlines() if l.strip()]
        header["Deliver To"] = "\n".join(lines)

    # Capture Conditions of purchase, Carriage terms, Settlement terms
    cond = re.search(r'Conditions of purchase:\s*(.*?)(?=\n|Carriage terms:)', text, re.I)
    carr = re.search(r'Carriage terms:\s*(.*?)(?=\n|Settlement terms:)', text, re.I)
    sett = re.search(r'Settlement terms:\s*(.*)', text, re.I)

    header["Conditions of purchase"] = cond.group(1).strip() if cond else None
    header["Carriage terms"] = carr.group(1).strip() if carr else None
    header["Settlement terms"] = sett.group(1).strip() if sett else None

    return header

def pdf_to_parts(text):
    parts = []
    matches = [m for m in PART_START.finditer(text)]
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        # Extract item number (3–5 digit before QUANTITY=)
        im = re.search(r'(\d{3,5})\s*\n\s*QUANTITY', block, re.I)
        item_no = im.group(1) if im else None

        # Extract item description (if available near DESCRIPTION=)
        desc = re.search(r'DESCRIPTION\s*=\s*(.*)', block, re.I)
        item_desc = desc.group(1).strip() if desc else None

        parts.append({
            "item_no": item_no,
            "item_desc": item_desc,
            "part_no": m.group(3).strip(),
            "text": block
        })
    return parts

# ------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Document Conversation", layout="wide")
st.title("Conversational RAG with Purchase Orders")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload Document")
    pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

    if pdf_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            try:
                # Parse PDF
                full_text = read_pdf_text(pdf_file)
                header = parse_header(full_text)
                parts = pdf_to_parts(full_text)

                documents = []
                for part in parts:
                    doc_text = f"""
                    HEADER INFO:
                    Deliver To: {header.get("Deliver To")}
                    Conditions of purchase: {header.get("Conditions of purchase")}
                    Carriage terms: {header.get("Carriage terms")}
                    Settlement terms: {header.get("Settlement terms")}

                    ITEM INFO:
                    Item No: {part.get("item_no")}
                    Item Desc: {part.get("item_desc")}
                    Part No: {part.get("part_no")}

                    FULL BLOCK:
                    {part['text']}
                    """
                    documents.append(doc_text)

                if not documents:
                    st.error("No parts extracted from PDF. Check the document format.")
                else:
                    # LangChain setup
                    embeddings = OllamaEmbeddings(model=Config.OLLAMA_EMBED_MODEL,
                                                  base_url=Config.OLLAMA_BASE_URL)
                    vector_store = FAISS.from_texts(texts=documents, embedding=embeddings)
                    retriever = vector_store.as_retriever()

                    llm = ChatOllama(model=Config.OLLAMA_CHAT_MODEL,
                                     base_url=Config.OLLAMA_BASE_URL,
                                     temperature=0.0)

                    memory = ConversationBufferMemory(memory_key="chat_history",
                                                      return_messages=True)

                    custom_prompt = PromptTemplate.from_template("""
                        You are a procurement assistant. Answer ONLY from the provided context.
                        - For 'Deliver To', always take it from HEADER INFO.
                        - If info is missing, reply: "Not found in the document."
                        Keep answers concise and factual.

                        Context: {context}
                        Question: {question}
                        Answer:
                    """)

                    st.session_state.chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=retriever,
                        memory=memory,
                        combine_docs_chain_kwargs={"prompt": custom_prompt}
                    )

                    st.session_state.messages = []
                    st.success("✅ Document processed! You can now ask questions.")

            except Exception as e:
                st.error(f"Error during processing: {e}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about the purchase order..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.chain is None:
        st.warning("Upload and process a document first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chain.invoke({"question": prompt})
                    answer = response.get("answer", "Sorry, could not find an answer.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error while answering: {e}")
