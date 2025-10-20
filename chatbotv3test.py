import os
import sqlite3
import logging
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# ---------------------- Load Environment ----------------------
load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------- Database Functions ----------------------
DB_PATH = "questions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS retrieved_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            source TEXT,
            page_number INTEGER,
            FOREIGN KEY(user_message_id) REFERENCES user_messages(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS llm_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message_id INTEGER NOT NULL,
            prompt_tokens INTEGER,
            response_tokens INTEGER,
            response_time REAL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(user_message_id) REFERENCES user_messages(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message_id INTEGER,
            satisfaction TEXT NOT NULL,
            feedback_text TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(user_message_id) REFERENCES user_messages(id)
        )
    """)
    conn.commit()
    conn.close()
    logging.info("📦 Database initialized successfully.")

def save_user_message(user_message, answer):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO user_messages (user_message, answer, timestamp)
        VALUES (?, ?, ?)
    """, (user_message, answer, datetime.now().isoformat()))
    message_id = c.lastrowid
    conn.commit()
    conn.close()
    logging.info("✅ Saved user message & answer to DB")
    return message_id

def save_retrieved_chunks(user_message_id, chunks):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for chunk in chunks:
        page_number = chunk.metadata.get("page_number", 0)
        c.execute("""
            INSERT INTO retrieved_chunks (user_message_id, chunk_text, source, page_number)
            VALUES (?, ?, ?, ?)
        """, (user_message_id, chunk.page_content, chunk.metadata.get("source"), page_number))
    conn.commit()
    conn.close()
    logging.info(f"📝 Saved {len(chunks)} chunks for user_message_id {user_message_id}")

def save_llm_metrics(user_message_id, prompt_tokens, response_tokens, response_time):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO llm_metrics (user_message_id, prompt_tokens, response_tokens, response_time, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_message_id, prompt_tokens, response_tokens, response_time, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    logging.info(f"📊 Saved LLM metrics for user_message_id {user_message_id}")

def save_feedback(user_message_id, satisfaction, feedback_text):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO feedback (user_message_id, satisfaction, feedback_text, timestamp)
        VALUES (?, ?, ?, ?)
    """, (user_message_id, satisfaction, feedback_text, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    logging.info(f"💬 Saved feedback for message {user_message_id}: {satisfaction}")

def count_tokens(text: str) -> int:
    return len(text.split())

# ---------------------- Load CSS ----------------------
def load_css():
    try:
        with open('styles.css', 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("styles.css not found. The app will run with default styling.")

# ---------------------- Streamlit Setup ----------------------
st.set_page_config(
    page_title="RAG Chatbot กยศ", 
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

# ---------------------- Header ----------------------
st.markdown("""
    <div class="chat-header">
        <h1 style="margin:0; font-size: 2.5rem;">🎓 RAG Chatbot กยศ</h1>
        <p style="margin:0.5rem 0 0 0; font-size: 1rem;">
            ระบบตอบคำถามอัจฉริยะเกี่ยวกับคุณสมบัติผู้กู้ยืมเงินกยศ
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.markdown("### 📚 ข้อมูลระบบ")
    st.markdown("""
    <div class='sidebar-info-box'>
        <p style='margin: 0; color: #374151;'><strong>🤖 Model:</strong> Llama 3.2</p>
        <p style='margin: 0.5rem 0; color: #374151;'><strong>📄 Embeddings:</strong> BGE-M3</p>
        <p style='margin: 0; color: #374151;'><strong>🔍 Retrieved Chunks:</strong> 3</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 วิธีใช้งาน")
    st.markdown("""
    <div style='color: #374151; line-height: 1.8;'>
        <ol style='color: #374151; padding-left: 1.2rem;'>
            <li style='margin: 0.5rem 0;'>พิมพ์คำถามในช่องแชท</li>
            <li style='margin: 0.5rem 0;'>รอระบบค้นหาและตอบคำถาม</li>
            <li style='margin: 0.5rem 0;'>กด "🛑 จบบทสนทนา" เมื่อต้องการจบ</li>
            <li style='margin: 0.5rem 0;'>ประเมินความพึงพอใจ</li>
            <li style='margin: 0.5rem 0;'>เริ่มแชทใหม่ได้ทันที</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.session_state.get("messages"):
        st.markdown("### 💬 สถิติการสนทนา")
        total_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f"""
        <div class='sidebar-info-box' style='text-align: center;'>
            <p style='margin: 0; color: #6b7280; font-size: 0.9rem;'>จำนวนคำถาม</p>
            <h2 style='margin: 0.5rem 0; color: #3b82f6;'>{total_msgs}</h2>
        </div>
        """, unsafe_allow_html=True)

# ---------------------- Initialize ----------------------
init_db()
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------- Load Document ----------------------
doc_path = "Loan_Features.pdf"

@st.cache_resource
def load_vectorstore():
    if not os.path.exists(doc_path):
        st.error(f"❌ ไม่พบไฟล์ {doc_path} กรุณาวางไฟล์ในตำแหน่งที่ถูกต้อง")
        st.stop()
    
    with st.spinner("📚 กำลังโหลดเอกสารและสร้าง Vector Database..."):
        loader = UnstructuredFileLoader(doc_path)
        docs_raw = loader.load()

        docs = []
        for d in docs_raw:
            source = d.metadata.get("source", doc_path)
            page_num = d.metadata.get("page_number", 1)
            docs.append(Document(page_content=d.page_content, metadata={"source": source, "page_number": page_num}))
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        logging.info(f"✅ Document split into {len(chunks)} chunks")
        
        embed = OllamaEmbeddings(model="bge-m3", base_url=OLLAMA_URL)
        persist_dir = "chroma_db_pdf"
        
        if os.path.exists(persist_dir):
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embed)
            logging.info("📂 Loaded existing ChromaDB")
        else:
            vectorstore = Chroma.from_documents(chunks, embed, persist_directory=persist_dir)
            logging.info("🆕 Created new ChromaDB and persisted")
        
        return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------- LLM Setup ----------------------
llm = OllamaLLM(model="llama3.2:latest", base_url=OLLAMA_URL, temperature=0.2)

template = """
คุณเป็นผู้ช่วย AI ที่เชี่ยวชาญด้านคุณสมบัติผู้กู้ยืมเงิน กยศ.
โปรดตอบคำถามอย่างชัดเจน กระชับ และเป็นมิตร ใช้ข้อมูลจากบริบทที่ให้มาเท่านั้น

บริบท (Context): {context}

คำถาม (Question): {question}

คำตอบ (Answer):
- ตอบเป็นภาษาไทยที่เข้าใจง่าย
- เจาะจงเกี่ยวกับคุณสมบัติผู้กู้ยืม กยศ.
- ถ้าไม่มีข้อมูลในบริบท ให้บอกว่า "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# ---------------------- Generate Answer ----------------------
def generate_answer(question: str):
    start_time = time.time()
    result = qa_chain({"query": question})
    end_time = time.time()

    answer_text = result["result"]
    retrieved_docs = result.get("source_documents", [])

    prompt_tokens = count_tokens(question)
    response_tokens = count_tokens(answer_text)
    response_time = end_time - start_time

    return answer_text, retrieved_docs, prompt_tokens, response_tokens, response_time

# ---------------------- Chat Interface ----------------------
if not st.session_state.get("chat_ended", False):
    if not st.session_state.messages:
        st.markdown("""
            <div class="info-box">
                <h3>👋 สวัสดีครับ! ยินดีต้อนรับสู่ระบบตอบคำถาม กยศ</h3>
                <p>คุณสามารถถามคำถามเกี่ยวกับคุณสมบัติผู้กู้ยืมเงิน กยศ ได้เลยครับ</p>
                <p><strong>ตัวอย่างคำถาม:</strong></p>
                <ul>
                    <li>ฉันต้องมีคุณสมบัติอย่างไรบ้างถึงจะกู้ กยศ ได้?</li>
                    <li>รายได้ครอบครัวต้องไม่เกินเท่าไหร่?</li>
                    <li>ฉันจบ ม.6 แล้ว กู้ กยศ ได้ไหม?</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ### >> FIX << ### แก้ไขส่วนแสดงประวัติแชทให้แสดงเลขหน้าด้วย
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "pages" in msg:
            st.caption(f"📄 อ้างอิงจากหน้า: {msg['pages']}")

if st.session_state.messages and not st.session_state.get("chat_ended", False):
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🛑 จบบทสนทนาและประเมินความพึงพอใจ", use_container_width=True):
            st.session_state.chat_ended = True
            st.rerun()

if not st.session_state.get("chat_ended", False):
    user_input = st.chat_input("💬 พิมพ์คำถามของคุณที่นี่...")
    
    if user_input:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 กำลังค้นหาข้อมูล..."):
                answer, retrieved_docs, prompt_tokens, response_tokens, response_time = generate_answer(user_input)
            
            st.markdown(answer)

            # ประมวลผลและแสดงเลขหน้าสำหรับคำตอบล่าสุด
            page_numbers_str = ""
            if retrieved_docs:
                unique_pages = sorted(list(set(doc.metadata.get("page_number", "N/A") for doc in retrieved_docs)))
                page_numbers_str = ", ".join(map(str, unique_pages))
                st.caption(f"📄 อ้างอิงจากหน้า: {page_numbers_str}")
            
            with st.expander("📊 ข้อมูลเพิ่มเติม"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Prompt Tokens", prompt_tokens)
                col2.metric("Response Tokens", response_tokens)
                col3.metric("Response Time", f"{response_time:.2f}s")
    
        # ### >> FIX << ### แก้ไขการบันทึก session state ให้เก็บเลขหน้าไปด้วย
        # Save to database
        user_message_id = save_user_message(user_input, answer)
        st.session_state.messages.append({"role": "user", "content": user_input, "id": user_message_id})
        save_retrieved_chunks(user_message_id, retrieved_docs)
        save_llm_metrics(user_message_id, prompt_tokens, response_tokens, response_time)

        # บันทึกข้อความของ assistant พร้อมเลขหน้า
        assistant_message = {"role": "assistant", "content": answer}
        if page_numbers_str:
            assistant_message["pages"] = page_numbers_str
        st.session_state.messages.append(assistant_message)
        
        st.rerun() 

# ---------------------- Feedback Section ----------------------
if st.session_state.get("chat_ended", False):
    st.markdown("""
        <div class="feedback-section">
            <h2 style="text-align: center; color: #3b82f6;">💬 แบบประเมินความพึงพอใจ</h2>
            <p style="text-align: center; color: #6b7280;">ความคิดเห็นของคุณช่วยให้เราพัฒนาระบบให้ดีขึ้นได้</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    satisfaction = st.radio(
        "🌟 คำตอบของแชทบอทมีประโยชน์หรือไม่?",
        ["ช่วยได้มาก 👍", "พอช่วยได้", "ยังไม่ช่วย 👎"],
        horizontal=True
    )
    
    feedback_text = st.text_area(
        "💭 ข้อเสนอแนะเพิ่มเติม (ไม่บังคับ):",
        placeholder="บอกเราได้เลยว่าเราควรปรับปรุงอะไรบ้าง...",
        height=100
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📩 ส่งฟีดแบค", type="primary", use_container_width=True):
            if satisfaction:
                last_user_msg_id = None
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "user":
                        last_user_msg_id = msg.get("id")
                        break
                
                if last_user_msg_id:
                    save_feedback(last_user_msg_id, satisfaction, feedback_text)
                    st.success("✅ ขอบคุณสำหรับความคิดเห็นของคุณ!")
                    time.sleep(2)
                    st.session_state.messages = []
                    st.session_state.chat_ended = False
                    st.rerun()
                else:
                     st.error("❌ ไม่พบข้อความผู้ใช้ล่าสุดเพื่อบันทึกฟีดแบค")
            else:
                st.warning("⚠️ กรุณาเลือกระดับความพึงพอใจก่อนส่ง")
    
    with col2:
        if st.button("🔄 เริ่มแชทใหม่", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_ended = False
            st.rerun()