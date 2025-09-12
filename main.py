import streamlit as st
import re
import json
import requests
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# --- CONFIGURATION ---
class Config:
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
    OLLAMA_CHAT_MODEL = "llama3.1:8b"


# --- PDF PARSING LOGIC ---
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
    # Capture block between Deliver to: and Invoice to:
    deliver = re.search(r'Deliver to:\s*(.*?)(?=Invoice to:)', text,
                        re.DOTALL | re.IGNORECASE)

    def clean_block(m):
        if not m:
            return None
        lines = [l.strip() for l in m.group(1).splitlines() if l.strip()]
        return " ".join(lines)  # or "\n".join(lines) for multi-line

    header = {"Deliver To": clean_block(deliver)}
    return header


def pdf_to_parts(text):
    parts = []
    matches = [m for m in PART_START.finditer(text)]
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        # Extract item number (4-digit code before QUANTITY=)
        im = re.search(r'\n\s*(\d{4})\s*\n.*?QUANTITY', block, re.S)
        item_no = im.group(1) if im else None

        parts.append({
            "item_no": item_no,
            "part_no": m.group(3).strip(),
            "text": block
        })
    return parts


# --- STREAMLIT APP ---

st.set_page_config(page_title="Document Conversation", layout="wide")
st.title("Conversational RAG with Purchase Orders")

# Initialize session state for messages and the chain
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload Document")
    pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

    if pdf_file and st.button("Process Document"):
        with st.spinner("Processing document... This may take a moment."):
            try:
                # 1. Read and Parse PDF
                full_text = read_pdf_text(pdf_file)
                header = parse_header(full_text)
                parts = pdf_to_parts(full_text)

                documents = []
                for part in parts:
                    doc_text = f"""
                    HEADER INFO:
                    Deliver To: {header.get("Deliver To")}

                    ITEM INFO:
                    Item No: {part.get("item_no")}
                    Part No: {part.get("part_no")}

                    FULL BLOCK:
                    {part['text']}
                    """
                    documents.append(doc_text)

                if not documents:
                    st.error("No parts could be extracted from the PDF. Please check the document format.")
                else:
                    # 2. Setup LangChain components
                    embeddings = OllamaEmbeddings(model=Config.OLLAMA_EMBED_MODEL, base_url=Config.OLLAMA_BASE_URL)
                    vector_store = FAISS.from_texts(texts=documents, embedding=embeddings)
                    retriever = vector_store.as_retriever()

                    llm = ChatOllama(model=Config.OLLAMA_CHAT_MODEL, base_url=Config.OLLAMA_BASE_URL, temperature=0.0)

                    # 3. Create Conversational Chain
                    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

                    custom_prompt = PromptTemplate.from_template("""
                        You are a procurement assistant. Answer the user's question based ONLY on the provided context. 
                        If the answer is not in the context, say "I cannot find this information in the document."
                        Always be concise and professional.
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

                    st.session_state.messages = []  # Clear previous chat
                    st.success("Document processed successfully! You can now ask questions.")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.chain is None:
        st.warning("Please upload and process a document first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chain.invoke({"question": prompt})
                    answer = response.get("answer", "Sorry, I encountered an issue and could not find an answer.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
