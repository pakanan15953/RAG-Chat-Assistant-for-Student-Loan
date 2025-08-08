import os
import torch
import streamlit as st
import logging
import sqlite3
import re
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
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
    
    # สร้างตารางใหม่หรืออัปเดตตารางเดิม
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    
    # ตรวจสอบและเพิ่มคอลัมน์ใหม่ถ้ายังไม่มี
    cursor.execute("PRAGMA table_info(questions)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'citations' not in columns:
        cursor.execute("ALTER TABLE questions ADD COLUMN citations TEXT DEFAULT ''")
        logging.info("✅ Added 'citations' column to database")
    
    if 'confidence_score' not in columns:
        cursor.execute("ALTER TABLE questions ADD COLUMN confidence_score REAL DEFAULT 0.0")
        logging.info("✅ Added 'confidence_score' column to database")
    
    conn.commit()
    conn.close()
    logging.info("📦 Database initialized successfully.")

def save_question_to_db(question, answer, citations="", confidence_score=0.0):
    conn = sqlite3.connect("questions.db")
    cursor = conn.cursor()
    
    # ตรวจสอบว่ามีคอลัมน์ citations และ confidence_score หรือไม่
    cursor.execute("PRAGMA table_info(questions)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'citations' in columns and 'confidence_score' in columns:
        # ใช้คำสั่ง SQL แบบเต็ม
        cursor.execute("""
            INSERT INTO questions (question, answer, citations, confidence_score, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (question, answer, citations, confidence_score, datetime.now().isoformat()))
    else:
        # ใช้คำสั่ง SQL แบบเดิม (backward compatibility)
        cursor.execute("""
            INSERT INTO questions (question, answer, timestamp)
            VALUES (?, ?, ?)
        """, (question, answer, datetime.now().isoformat()))
        logging.warning("⚠️ Using legacy database format (missing citations/confidence columns)")
    
    conn.commit()
    conn.close()
    logging.info("✅ Saved question to database.")

# ---------------------- Citation System ----------------------
class CitationSystem:
    def __init__(self):
        self.citation_counter = 0
        self.citations = {}
    
    def add_citation(self, doc_content, page_num, chunk_id, similarity_score):
        """เพิ่มการอ้างอิงใหม่"""
        self.citation_counter += 1
        citation_key = f"ref_{self.citation_counter}"
        
        # แก้ไขปัญหาหน้าที่ไม่ทราบ
        if page_num is None or page_num == "ไม่ทราบหน้า":
            # พยายามหาหน้าจาก metadata อื่น
            if hasattr(doc_content, 'metadata'):
                page_num = doc_content.metadata.get('page', None)
            
            # ถ้าไม่มีเลย ให้ใช้ chunk_id เป็นตัวบอก
            if page_num is None:
                page_num = f"ส่วนที่ {chunk_id.replace('chunk_', '')}"
        
        # หาบรรทัดที่เกี่ยวข้อง (ประมาณการ)
        lines = doc_content.split('\n') if isinstance(doc_content, str) else str(doc_content).split('\n')
        total_chars = len(str(doc_content))
        lines_info = []
        
        for i, line in enumerate(lines, 1):
            if line.strip():  # ข้ามบรรทัดเปล่า
                lines_info.append(f"บรรทัด {i}")
        
        self.citations[citation_key] = {
            'page': page_num,
            'content': str(doc_content)[:200] + "..." if len(str(doc_content)) > 200 else str(doc_content),
            'full_content': str(doc_content),
            'chunk_id': chunk_id,
            'similarity_score': round(similarity_score, 3),
            'lines_range': f"{lines_info[0]}-{lines_info[-1]}" if lines_info else "ส่วนที่ 1",
            'confidence': self.calculate_confidence(similarity_score, len(str(doc_content)))
        }
        
        return citation_key
    
    def calculate_confidence(self, similarity_score, content_length):
        """คำนวณความเชื่อมั่นจากคะแนนความคล้ายและความยาวเนื้อหา"""
        base_confidence = similarity_score * 100
        
        # ปรับตามความยาวเนื้อหา
        if content_length > 500:
            length_bonus = 5
        elif content_length > 200:
            length_bonus = 3
        else:
            length_bonus = 0
            
        return min(95, base_confidence + length_bonus)
    
    def format_citation_inline(self, citation_key):
        """สร้างการอ้างอิงแบบ inline"""
        if citation_key in self.citations:
            cite = self.citations[citation_key]
            return f"[หน้า {cite['page']}, {cite['lines_range']}]"
        return "[?]"
    
    def format_citation_detailed(self, citation_key):
        """สร้างการอ้างอิงแบบละเอียด"""
        if citation_key in self.citations:
            cite = self.citations[citation_key]
            return f"""
**อ้างอิง {citation_key}:**
- 📄 **หน้า:** {cite['page']}
- 📍 **ตำแหน่ง:** {cite['lines_range']} 
- 📊 **ความเชื่อมั่น:** {cite['confidence']:.1f}%
- 🎯 **ความตรงประเด็น:** {cite['similarity_score']:.3f}
- 📝 **เนื้อหา:** "{cite['content']}"
"""
        return "ไม่พบข้อมูลอ้างอิง"
    
    def get_all_citations_summary(self):
        """สรุปการอ้างอิงทั้งหมด"""
        if not self.citations:
            return "ไม่มีการอ้างอิง"
        
        summary = "## 📚 แหล่งข้อมูลที่ใช้อ้างอิง\n\n"
        for key, cite in self.citations.items():
            summary += f"**{key}:** หน้า {cite['page']}, {cite['lines_range']} (ความเชื่อมั่น: {cite['confidence']:.1f}%)\n"
        
        return summary
    
    def get_average_confidence(self):
        """คำนวณค่าเฉลี่ยความเชื่อมั่น"""
        if not self.citations:
            return 0
        
        total_confidence = sum(cite['confidence'] for cite in self.citations.values())
        return total_confidence / len(self.citations)

# ---------------------- Embedding Setup ----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
logging.info("💡 Embedding model loaded")

# ---------------------- Vector Store ----------------------
persist_directory = "chroma_db"
if os.path.exists(persist_directory):
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    logging.info("📦 Loaded existing vector store from disk")
else:
    logging.info("📄 Processing document for the first time...")
    loader = PyMuPDFLoader("Loan_Features.pdf")
    docs = loader.load()

    if not docs or not docs[0].page_content.strip():
        logging.error("❌ เอกสารไม่มีเนื้อหา หรือโหลดไม่สำเร็จ")
        st.error("❌ เอกสารไม่มีเนื้อหา หรือโหลดไม่สำเร็จ")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    logging.info(f"✅ Document split into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    vectorstore.persist()
    logging.info("📚 Vector store created and persisted successfully")

# ---------------------- Enhanced Retrieval ----------------------
def retrieve_with_citations(query: str):
    """ค้นหาเอกสารพร้อมสร้างระบบอ้างอิง"""
    # ค้นหาด้วยคะแนนความคล้าย
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
    
    citation_system = CitationSystem()
    context_parts = []
    
    # กรองเฉพาะผลลัพธ์ที่มีคะแนนดี (< 0.7 คือดี, > 1.0 คือแย่)
    filtered_docs = [(doc, score) for doc, score in docs_with_scores if score < 0.8]
    
    if not filtered_docs:
        # ถ้าไม่มีผลลัพธ์ที่ดี ให้เอาอันที่ดีที่สุด 2 อัน
        filtered_docs = docs_with_scores[:2]
    
    for i, (doc, score) in enumerate(filtered_docs[:3]):  # จำกัดแค่ 3 อันดับแรก
        # ปรับปรุงการดึงหมายเลขหน้า
        page = None
        
        # วิธีที่ 1: ดึงจาก metadata
        if hasattr(doc, 'metadata') and doc.metadata:
            page = doc.metadata.get("page", None)
            # บางครั้ง page อาจเป็น 0-based ต้องบวก 1
            if isinstance(page, int):
                page = page + 1 if page >= 0 else 1
        
        # วิธีที่ 2: หาจากเนื้อหา (ถ้ามี pattern)
        if page is None:
            import re
            content = str(doc.page_content)
            # หา pattern เหมือน "หน้า 5" หรือ "Page 5"
            page_match = re.search(r'(?:หน้า|Page)\s*(\d+)', content, re.IGNORECASE)
            if page_match:
                page = int(page_match.group(1))
        
        # วิธีที่ 3: ใช้ลำดับ chunk แทน
        if page is None:
            page = f"ส่วนที่ {i+1}"
        
        # Debug log
        logging.info(f"📄 เอกสารที่ {i+1}: หน้า {page}, คะแนน: {score:.3f}")
        logging.info(f"Metadata: {doc.metadata if hasattr(doc, 'metadata') else 'No metadata'}")
        logging.info(f"เนื้อหา: {str(doc.page_content)[:200]}...\n")
        
        # สร้างการอ้างอิง
        citation_key = citation_system.add_citation(
            doc.page_content, 
            page, 
            f"chunk_{i+1}", 
            1 - score  # แปลงเป็น similarity score (ยิ่งใกล้ 1 ยิ่งคล้าย)
        )
        
        # สร้าง context พร้อมการอ้างอิง
        citation_inline = citation_system.format_citation_inline(citation_key)
        context_parts.append(f"{doc.page_content} {citation_inline}")
    
    context = "\n\n".join(context_parts)
    return context, citation_system

# ---------------------- Enhanced Answer Generation ----------------------
def generate_answer_with_citations(query: str, context: str, citation_system: CitationSystem) -> tuple:
    """สร้างคำตอบพร้อมระบบอ้างอิง"""
    
    messages = [
        {
            "role": "system",
            "content": 
                "คุณเป็น AI ผู้ช่วยในการตอบคำถามเกี่ยวกับคุณสมบัติผู้กู้ยืมกยศ "
                "ตอบคำถามให้ถูกต้องและอ้างอิงจากบริบทที่ให้มาเท่านั้น "
                "ระบุหน้าและบรรทัดที่อ้างอิงในคำตอบด้วย "
                "เขียนลงท้ายด้วยคำว่า 'ครับ' เสมอ"
        },
        {
            "role": "user",
            "content": (
                "Context พร้อมการอ้างอิง:\n"
                f"{context}\n\n"
                "ตัวอย่างการตอบพร้อมอ้างอิง:\n"
                "Q: รายได้ครอบครัวของผู้กู้ต้องไม่เกินเท่าไหร่ต่อปี?\n"
                "A: รายได้ครอบครัวของผู้กู้ต้องไม่เกิน 360,000 บาทต่อปี [หน้า 5, บรรทัด 12-15] ครับ\n\n"
                f"Q: {query}\n"
                "A:"
            )
        }
    ]
    
    response = chat(model="llama3.2:latest", messages=messages)
    answer = response["message"]["content"]
    
    # คำนวณความเชื่อมั่นรวม
    avg_confidence = citation_system.get_average_confidence()
    
    return answer, avg_confidence

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="RAG Chatbot กยศ - Enhanced", page_icon="📄", layout="wide")

# Header
st.title("📄 RAG Chatbot กยศ - Enhanced Citation System")
st.write("ถามคำถามจากเอกสาร `Loan_Features.pdf` พร้อมระบบอ้างอิงที่ละเอียด")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    show_citations = st.checkbox("แสดงการอ้างอิงแบบละเอียด", value=True)
    show_confidence = st.checkbox("แสดงคะแนนความเชื่อมั่น", value=True)
    
    st.header("📊 สถิติ")
    if st.button("ดูประวัติคำถาม"):
        conn = sqlite3.connect("questions.db")
        df = st.experimental_get_query_params() # placeholder for actual query
        st.write("ฟีเจอร์นี้อยู่ระหว่างพัฒนา")

# Initialize
init_db()

# Setup chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main chat area
col1, col2 = st.columns([2, 1]) if show_citations else st.columns([1])

with col1:
    # Show previous conversation
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # แสดงความเชื่อมั่นถ้าเป็นคำตอบของ AI
            if msg["role"] == "assistant" and show_confidence and "confidence" in msg:
                confidence_color = "green" if msg["confidence"] > 80 else "orange" if msg["confidence"] > 60 else "red"
                st.markdown(f"<span style='color: {confidence_color}'>🎯 ความเชื่อมั่น: {msg['confidence']:.1f}%</span>", unsafe_allow_html=True)

    # Chat input
    user_query = st.chat_input("❓ ถามคำถามเกี่ยวกับคุณสมบัติผู้กู้ยืมกยศ:")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("📚 กำลังค้นหาข้อมูลที่เกี่ยวข้อง..."):
            context, citation_system = retrieve_with_citations(user_query)

        with st.spinner("🧠 กำลังสร้างคำตอบ..."):
            answer, confidence = generate_answer_with_citations(user_query, context, citation_system)

        # เก็บคำตอบพร้อมข้อมูลเพิ่มเติม
        assistant_msg = {
            "role": "assistant", 
            "content": answer,
            "confidence": confidence,
            "citation_system": citation_system
        }
        
        st.session_state.chat_history.append(assistant_msg)
        
        with st.chat_message("assistant"):
            st.markdown(answer)
            
            if show_confidence:
                confidence_color = "green" if confidence > 80 else "orange" if confidence > 60 else "red"
                st.markdown(f"<span style='color: {confidence_color}'>🎯 ความเชื่อมั่น: {confidence:.1f}%</span>", unsafe_allow_html=True)

        # บันทึกลงฐานข้อมูล
        citations_summary = citation_system.get_all_citations_summary()
        save_question_to_db(user_query, answer, citations_summary, confidence)

# Citation panel
if show_citations and len(st.session_state.chat_history) > 0:
    with col2:
        st.header("📚 การอ้างอิง")
        
        # แสดงการอ้างอิงของคำตอบล่าสุด
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
            last_msg = st.session_state.chat_history[-1]
            if "citation_system" in last_msg:
                citation_system = last_msg["citation_system"]
                
                st.markdown("### แหล่งข้อมูลที่ใช้:")
                for key, cite in citation_system.citations.items():
                    with st.expander(f"📄 หน้า {cite['page']} (ความเชื่อมั่น: {cite['confidence']:.1f}%)"):
                        st.markdown(f"**ตำแหน่ง:** {cite['lines_range']}")
                        st.markdown(f"**ความตรงประเด็น:** {cite['similarity_score']:.3f}")
                        st.markdown("**เนื้อหา:**")
                        st.text_area("", cite['full_content'], height=150, key=f"content_{key}")

# Footer
st.markdown("---")
st.markdown("💡 **คำแนะนำ:** ระบบจะแสดงแหล่งที่มาของข้อมูลทุกครั้งที่ตอบคำถาม เพื่อให้คุณสามารถตรวจสอบความถูกต้องได้")