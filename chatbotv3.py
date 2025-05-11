import os
import torch
import streamlit as st
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # อัปเดตจาก langchain -> langchain_community
from langchain_community.vectorstores import Chroma  # อัปเดตจาก langchain -> langchain_community
from ollama import chat

# 1. Load document (.docx)
loader = UnstructuredFileLoader("Loan_Features.docx")
docs = loader.load()

# ตรวจสอบว่าโหลดสำเร็จ
if not docs or not docs[0].page_content.strip():
    st.error("❌ เอกสารไม่มีเนื้อหา หรือโหลดไม่สำเร็จ")
    st.stop()

# 2. แบ่งเนื้อหาเป็น chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. ใช้ BGE Embedding จาก HuggingFace
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 4. สร้าง vector store จาก chunks
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# 5. ดึงข้อมูลที่เกี่ยวข้อง
def retrieve(query: str):
    return vectorstore.similarity_search(query, k=3)

# 6. ใช้ Ollama ในการสร้างคำตอบ
def generate_answer(query: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "คุณเป็น ai ผู้ช่วยในการตอบคำถามเกี่ยวกับคุณสมบัติผู้กู้ยืมกยศ ตอบคำถามให้ถูกต้องและอ้างอิงจากบริบทที่ให้มาเท่านั้น และตอบลงท้ายด้วย ค่ะ/ครับ"
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]
    response = chat(model="llama3.2:latest", messages=messages)
    return response["message"]["content"]

# 7. Streamlit UI
st.set_page_config(page_title="RAG Chatbot กยศ", page_icon="📄")
st.title("📄 RAG Chatbot กยศ ด้วย BGE + Ollama")
st.write("ถามคำถามจากเอกสาร `Loan_Features.docx`")

# รับคำถามจากผู้ใช้
user_query = st.chat_input("❓ ถามคำถามเกี่ยวกับคุณสมบัติผู้กู้ยืมกยศ:")

if user_query:
    with st.spinner("📚 กำลังค้นหาข้อมูลที่เกี่ยวข้อง..."):
        retrieved_docs = retrieve(user_query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    with st.spinner("🧠 กำลังสร้างคำตอบ..."):
        answer = generate_answer(user_query, context)

    # แสดงคำตอบ
    st.markdown("### ✅ คำตอบ")
    st.markdown(user_query)
    st.markdown(answer)

    # แสดงอ้างอิง
    ##st.markdown("### 🔎 อ้างอิงจากเอกสาร:")
    ##for i, doc in enumerate(retrieved_docs, 1):
      ##  source = doc.metadata.get("source", "ไม่ทราบที่มา")
      ##  snippet = doc.page_content[:300].strip().replace("\n", " ")
      ##  st.markdown(f"**{i}.** `{source}`")
       ## st.markdown(f"> {snippet}...")

