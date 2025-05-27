import os
import torch
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ollama import chat

# 1. Load document (.docx)
loader = UnstructuredFileLoader("Loan_Features.docx")
docs = loader.load()

if not docs or not docs[0].page_content.strip():
    print("❌ เอกสารไม่มีเนื้อหา หรือโหลดไม่สำเร็จ")
    exit()

# 2. แบ่งเนื้อหาเป็น chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 4. Vectorstore
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# 5. ดึงข้อมูลที่เกี่ยวข้อง
def retrieve(query: str):
    return vectorstore.similarity_search(query, k=3)

# 6. สร้างคำตอบ
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

# 👇 รับคำถามจากผู้ใช้ผ่าน terminal
user_query = input("📌 พิมพ์คำถามเกี่ยวกับผู้กู้ กยศ: ")

related_docs = retrieve(user_query)
context = "\n\n".join([doc.page_content for doc in related_docs])
answer = generate_answer(user_query, context)

print("\n✅ คำตอบ:")
print(answer)
