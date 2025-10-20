import shutil

persist_dir = "chroma_db"

# ลบ persist directory เก่า
shutil.rmtree(persist_dir, ignore_errors=True)

# สร้าง DB ใหม่
vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory=persist_dir
)

# เพิ่มเอกสารลง DB
vectorstore.add_documents(chunks)
vectorstore.persist()
