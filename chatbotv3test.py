import os
import torch
import streamlit as st
import logging
import sqlite3
from datetime import datetime
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ollama import chat
from transformers import pipeline

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------- Database ----------------------
def init_db():
    conn = sqlite3.connect("questions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            like INTEGER DEFAULT 0,
            unlike INTEGER DEFAULT 0
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
    question_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logging.info("✅ Saved question to database.")
    return question_id

def save_feedback_to_db(question_id: int, feedback: str):
    conn = sqlite3.connect("questions.db")
    cursor = conn.cursor()
    if feedback == "like":
        cursor.execute("UPDATE questions SET like = 1 WHERE id = ?", (question_id,))
    elif feedback == "unlike":
        cursor.execute("UPDATE questions SET unlike = 1 WHERE id = ?", (question_id,))
    conn.commit()
    conn.close()
    logging.info(f"📝 Feedback saved: {feedback} for question_id {question_id}")

# ---------------------- Intent Classification ----------------------
st.set_page_config(page_title="RAG Chatbot กยศ", page_icon="📄")
st.write("💡 กำลังโหลดโมเดล Intent Classification...")
classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

intent_map = {
    "joy": "GREETING",
    "love": "GREETING",
    "surprise": "GREETING",
    "sadness": "QUESTION",
    "anger": "QUESTION",
    "fear": "QUESTION"
}

def classify_intent(text: str) -> str:
    result = classifier(text)[0]
    label = result['label']
    return intent_map.get(label, "OTHER")

# ---------------------- Keyword Filter ----------------------
GREETING_KEYWORDS = [
    "สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ", "hello", "hi", "hey", "หวัดดี",
    "good morning", "good afternoon", "good evening",
    "ขอบคุณ", "ขอบคุณครับ", "ขอบคุณค่ะ",
    "ขอโทษ", "ขอโทษครับ", "ขอโทษค่ะ",
    "โอเค", "ok", "เยี่ยม", "ดีมาก", "great",
    "ฮ่า", "ฮ่า ๆ", "lol", "haha", "hehe",
    "ใช่", "ไม่ใช่", "ครับ", "ค่ะ", "เออ", "อืม", "จริงหรือ", "จริงเหรอ"
]

def should_store_question(text: str) -> bool:
    if any(word.lower() in text.lower() for word in GREETING_KEYWORDS):
        return False
    intent = classify_intent(text)
    if intent in ["GREETING", "OTHER"]:
        return False
    return True

# ---------------------- RAG Pipeline ----------------------
st.write("📄 กำลังโหลดเอกสาร...")
loader = UnstructuredFileLoader("Loan_Features.docx")
docs = loader.load()

if not docs or not docs[0].page_content.strip():
    st.error("❌ เอกสารไม่มีเนื้อหา หรือโหลดไม่สำเร็จ")
    st.stop()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
logging.info(f"✅ Document split into {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
logging.info("💡 Embedding model loaded")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
logging.info("📚 Vector store created successfully")

def retrieve(query: str):
    return vectorstore.similarity_search(query, k=3)

def generate_answer(user_query: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "คุณเป็น AI ผู้ช่วยในการตอบคำถามเกี่ยวกับคุณสมบัติผู้กู้ยืมกยศ "
                "ตอบคำถามให้ถูกต้องและอ้างอิงจากบริบทที่ให้มาเท่านั้น "
                "เขียนลงท้ายด้วยคำว่า ครับ"
            )
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
    return response["message"]["content"]

# ---------------------- Streamlit Chat UI ----------------------
st.title("📄 RAG Chatbot กยศ (ChatGPT-style)")

init_db()

# เก็บแชททั้งหมด
if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_ids" not in st.session_state:
    st.session_state.question_ids = []

# ตัวแปรเก็บสถานะให้คะแนน (ครั้งเดียว)
if "has_rated" not in st.session_state:
    st.session_state.has_rated = False

user_input = st.chat_input("❓ ถามคำถามเกี่ยวกับคุณสมบัติผู้กู้ยืมกยศ:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("📚 กำลังค้นหาข้อมูลที่เกี่ยวข้อง..."):
        retrieved_docs = retrieve(user_input)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        logging.info("🔍 Retrieved relevant context")
    
    with st.spinner("🧠 กำลังสร้างคำตอบ..."):
        answer = generate_answer(user_input, context)
        logging.info("✅ Answer generated")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    question_id = None
    if should_store_question(user_input):
        question_id = save_question_to_db(user_input, answer)
        st.session_state.question_ids.append(question_id)
        st.success("✅ บันทึกคำถามเรียบร้อย")
    else:
        st.info("ℹ️ คำถามนี้ไม่เก็บ เพราะเป็น Greeting / Small Talk")

# แสดงแชททั้งหมดแบบ ChatGPT พร้อมปุ่ม Like/Unlike
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            
            # แสดงปุ่ม Like/Unlike เฉพาะเมื่อยังไม่เคยให้คะแนน
            if not st.session_state.has_rated:
                if idx > 0 and idx-1 < len(st.session_state.question_ids):
                    question_id = st.session_state.question_ids[idx-1]
                    col1, col2 = st.columns(2)

                    if col1.button("👍 Like", key=f"like_{idx}"):
                        save_feedback_to_db(question_id, "like")
                        st.session_state.has_rated = True  # ปุ่มจะหายทันที
                        st.success("ขอบคุณสำหรับการให้คะแนน 👍")
                    if col2.button("👎 Unlike", key=f"unlike_{idx}"):
                        save_feedback_to_db(question_id, "unlike")
                        st.session_state.has_rated = True  # ปุ่มจะหายทันที
                        st.warning("รับทราบการให้คะแนน 👎")
            else:
                st.info("คุณได้ให้คะแนนแล้ว ✅")
