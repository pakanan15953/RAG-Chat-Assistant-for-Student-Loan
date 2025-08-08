import os
import torch
import logging
import sqlite3
import gradio as gr
from datetime import datetime
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
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

if not docs or not docs[0].page_content.strip():
    raise ValueError("❌ เอกสารไม่มีเนื้อหา หรือโหลดไม่สำเร็จ")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
logging.info(f"✅ Document split into {len(chunks)} chunks")

# 3. Load Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
logging.info("💡 Embedding model loaded")

# 4. Create Vector Store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
logging.info("📚 Vector store created successfully")

# 5. Retrieval Function
def retrieve(query: str):
    return vectorstore.similarity_search(query, k=3)

# 6. Answer Generation with Ollama
def generate_answer(user_query: str):
    retrieved_docs = retrieve(user_query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    messages = [
        {
            "role": "system",
            "content": "คุณเป็น ai ผู้ช่วยในการตอบคำถามเกี่ยวกับคุณสมบัติผู้กู้ยืมกยศ ตอบคำถามให้ถูกต้องและอ้างอิงจากบริบทที่ให้มาเท่านั้น เขียนลงท้ายด้วยคำว่า ครับ "
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                "ตัวอย่างคำถาม-คำตอบ:\n"
                "Q: รายได้ครอบครัวของผู้กู้ต้องไม่เกินเท่าไหร่ต่อปี?\n"
                "A: ไม่เกิน 360,000 บาทต่อปีครับ\n"
                "Q: คนอายุ 33 ปี ยังสามารถกู้ได้หรือไม่?\n"
                "A: ได้ครับ เฉพาะลักษณะที่ 4 (จำกัดอายุไม่เกิน 35 ปี)\n\n"
                f"Q: {user_query}\nA:"
            )
        }
    ]
    response = chat(model="llama3.2:latest", messages=messages)
    answer = response["message"]["content"]
    save_question_to_db(user_query, answer)
    return answer

# ---------------------- Gradio UI ----------------------
init_db()

demo = gr.Interface(
    fn=generate_answer,
    inputs=gr.Textbox(label="❓ ถามคำถามเกี่ยวกับคุณสมบัติผู้กู้ยืมกยศ"),
    outputs=gr.Textbox(label="✅ คำตอบ"),
    title="📄 RAG Chatbot กยศ",
    description="ระบบตอบคำถามอัตโนมัติจากเอกสาร `Loan_Features.docx` โดยใช้ RAG + Ollama",
    allow_flagging="never"
)

demo.launch()