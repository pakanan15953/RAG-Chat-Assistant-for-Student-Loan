import os
import torch
import streamlit as st
import logging
import sqlite3
from datetime import datetime
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from ollama import chat

# ---------------------- Logging Setup ----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
# 1. Load document
logging.info("📄 Loading document...")
loader = UnstructuredFileLoader("Loan_Features.docx")
docs = loader.load()

# Check if document loaded properly
if not docs or not docs[0].page_content.strip():
    logging.error("❌ เอกสารไม่มีเนื้อหา หรือโหลดไม่สำเร็จ")
    st.error("❌ เอกสารไม่มีเนื้อหา หรือโหลดไม่สำเร็จ")
    st.stop()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
logging.info(f"✅ Document split into {len(chunks)} chunks")

# 3. Use HuggingFace multilingual embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"},  # บังคับใช้ CPU เท่านั้น
    encode_kwargs={"normalize_embeddings": True}
)
logging.info("💡 Embedding model loaded")

# 4. Create vector store
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
logging.info("📚 Vector store created successfully")

# 5. Retrieval function
def retrieve(query: str):
    return vectorstore.similarity_search(query, k=3)

# 6. Answer generation with Ollama
def generate_answer(query: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "คุณเป็น ai ผู้ช่วยในการตอบคำถามเกี่ยวกับคุณสมบัติผู้กู้ยืมกยศ ตอบคำถามให้ถูกต้องและอ้างอิงจากบริบทที่ให้มาเท่านั้น เขียนลงท้ายด้วยคำว่า ครับ "
        },
        {
        "role": "user",
        "content": (
            "Context:\n"
            f"{context}\n\n"
            "ตัวอย่างคำถาม-คำตอบ:\n"
            "Q: รายได้ครอบครัวของผู้กู้ต้องไม่เกินเท่าไหร่ต่อปี?\n"
            "A: ไม่เกิน 360,000 บาทต่อปีครับ\n"
            "Q: คนอายุ 33 ปี ยังสามารถกู้ได้หรือไม่?\n"
            "A: ได้ครับ เฉพาะลักษณะที่ 4 (จำกัดอายุไม่เกิน 35 ปี)\n\n"
            f"Q: {user_query}\n"
            "A:"
        )
    }
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
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        logging.info("🔍 Retrieved relevant context")

    with st.spinner("🧠 กำลังสร้างคำตอบ..."):
        answer = generate_answer(user_query, context)
        logging.info("✅ Answer generated")

    # Show result
    st.markdown(user_query)
    st.markdown(" ✅ คำตอบ")
    st.markdown(answer)

    # Save to DB
    save_question_to_db(user_query, answer)


