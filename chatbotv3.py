import os
import torch
import streamlit as st
import logging
import sqlite3
from datetime import datetime
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ollama import chat

# ---------------------- Logging Setup ----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# ---------------------- Database Setup ----------------------
def init_db():
    conn = sqlite3.connect("questions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logging.info("📦 Database initialized successfully.")

def save_question_to_db(question, answer):
    conn = sqlite3.connect("questions.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO questions (question, answer, timestamp)
        VALUES (?, ?, ?)
    """, (question, answer, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    logging.info("✅ Saved question to database.")

# ---------------------- RAG Pipeline ----------------------
logging.info("📄 Loading document...")
loader = UnstructuredFileLoader("Loan_Features.docx")
docs = loader.load()

if not docs or not docs[0].page_content.strip():
    logging.error("❌ เอกสารไม่มีเนื้อหา หรือโหลดไม่สำเร็จ")
    st.error("❌ เอกสารไม่มีเนื้อหา หรือโหลดไม่สำเร็จ")
    st.stop()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
logging.info(f"✅ Document split into {len(chunks)} chunks")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
logging.info("💡 Embedding model loaded")

# Vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
logging.info("📚 Vector store created successfully")

# ---------------------- Retrieval with confidence ----------------------
def retrieve(query: str):
    docs_with_score = vectorstore.similarity_search_with_score(query, k=5)
    
    # แปลง score เป็น similarity (ยิ่งใกล้ 1 ยิ่งเหมือน)
    filtered = [(doc, 1 - score) for doc, score in docs_with_score if score < 0.8]

    # top3
    top3 = filtered[:3]

    results = []
    for rank, (doc, similarity) in enumerate(top3, 1):
        # top1 confidence สูงสุด, top2 ลดลง, top3 ลดลงอีก
        confidence = similarity * 90 + (3 - rank) * 3  # top1+6, top2+3, top3+0
        results.append({"doc": doc, "confidence": confidence})
    
    return results

# ---------------------- Answer generation ----------------------
def generate_answer(query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "คุณเป็น ai ผู้ช่วยตอบคำถามเกี่ยวกับคุณสมบัติผู้กู้ยืมกยศ ตอบจาก context ที่ให้เท่านั้น ลงท้ายคำตอบด้วยคำว่า ครับ"},
        {"role": "user", "content": f"Context:\n{context}\n\nQ: {query}\nA:"}
    ]
    response = chat(model="llama3.2:latest", messages=messages)
    return response["message"]["content"]

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="RAG Chatbot กยศ", page_icon="📄")
st.title("📄 RAG Chatbot กยศ")
st.write("ถามคำถามจากเอกสาร `Loan_Features.docx`")

# Initialize DB
init_db()

# User input
user_query = st.chat_input("❓ ถามคำถามเกี่ยวกับคุณสมบัติผู้กู้ยืมกยศ:")

if user_query:
    with st.spinner("📚 กำลังค้นหาข้อมูลที่เกี่ยวข้อง..."):
        retrieved_docs = retrieve(user_query)
        context_parts = []
        for i, item in enumerate(retrieved_docs, 1):
            context_parts.append(f"{item['doc'].page_content} [Confidence: {item['confidence']:.1f}%]")
        context = "\n\n".join(context_parts)
        logging.info("🔍 Retrieved relevant context with confidence")

    with st.spinner("🧠 กำลังสร้างคำตอบ..."):
        answer = generate_answer(user_query, context)
        logging.info("✅ Answer generated")

    # Show result
    st.markdown(f"**คำถาม:** {user_query}")
    st.markdown("**คำตอบ:**")
    st.markdown(answer)

    # Save to DB
    save_question_to_db(user_query, answer)
