import sqlite3
import streamlit as st
import pandas as pd

DB_PATH = "questions.db"

# โหลดข้อมูลจาก SQLite
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM questions ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# ลบข้อมูลทั้งหมด
def delete_all_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM questions")
    conn.commit()
    conn.close()

# บันทึกคำตอบที่ถูกต้อง
def update_correct_answer(question_id, correct_answer):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE questions SET correct_answer = ? WHERE id = ?", (correct_answer, question_id))
    conn.commit()
    conn.close()

# UI
st.set_page_config(page_title="Admin Dashboard - คำถามทั้งหมด", page_icon="🧑‍💼")
st.title("🧑‍💼 Admin Dashboard")
st.write("แสดงคำถามที่ถูกถาม พร้อมแก้ไขคำตอบที่ถูกต้องได้")

df = load_data()

if df.empty:
    st.info("ยังไม่มีข้อมูลคำถามในระบบ")
else:
    # แสดงตารางแบบ editable เฉพาะคอลัมน์ correct_answer
    editable_df = df[["id", "question", "answer", "correct_answer", "timestamp"]].copy()
    edited_df = st.data_editor(
        editable_df,
        column_config={
            "question": "คำถาม",
            "answer": "คำตอบที่ AI ตอบ",
            "correct_answer": st.column_config.TextColumn("คำตอบที่ถูกต้อง (Editable)"),
            "timestamp": "เวลาที่ถาม"
        },
        disabled=["id", "question", "answer", "timestamp"],
        use_container_width=True,
        num_rows="dynamic"
    )

    if st.button("💾 บันทึกคำตอบที่ถูกต้องทั้งหมด"):
        changes = 0
        for i in range(len(df)):
            old = df.loc[i, "correct_answer"]
            new = edited_df.loc[i, "correct_answer"]
            if old != new:
                update_correct_answer(df.loc[i, "id"], new)
                changes += 1
        if changes > 0:
            st.success(f"✅ บันทึกคำตอบที่ถูกต้องแล้ว {changes} รายการ")
        else:
            st.info("ไม่มีการเปลี่ยนแปลงข้อมูล")
        st.rerun()

    with st.expander("🗑️ ลบข้อมูลทั้งหมด"):
        if st.button("⚠️ ยืนยันลบข้อมูลทั้งหมด"):
            delete_all_data()
            st.success("ลบข้อมูลทั้งหมดเรียบร้อยแล้ว")
            st.rerun()
