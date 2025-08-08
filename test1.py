import streamlit as st
import sqlite3
import pandas as pd
import bcrypt
from datetime import datetime

# --- ตั้งค่าฐานข้อมูลผู้ใช้ (secure_users.db) ---
user_conn = sqlite3.connect('secure_users.db', check_same_thread=False)
user_c = user_conn.cursor()

user_c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')
user_conn.commit()

# --- ฟังก์ชันสำหรับผู้ใช้ ---
def add_user(username, plain_password):
    hashed_pw = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())
    try:
        user_c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        user_conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username, password):
    user_c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = user_c.fetchone()
    if result:
        hashed_pw = result[0]
        return bcrypt.checkpw(password.encode('utf-8'), hashed_pw)
    return False

# ✅ คุณสามารถเพิ่ม user ได้โดยรันคำสั่งนี้ครั้งเดียว แล้วคอมเมนต์ออก
#add_user("glueta", "1234")
#add_user("fasai", "1234")
# --- ตั้งค่าฐานข้อมูลคำถาม (questions.db) ---
qa_conn = sqlite3.connect('questions.db', check_same_thread=False)
qa_c = qa_conn.cursor()

qa_c.execute('''
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    correct_answer TEXT
)
''')
qa_conn.commit()

def get_all_questions():
    qa_c.execute("SELECT * FROM questions")
    return qa_c.fetchall()

def update_correct_answer(id, new_answer):
    qa_c.execute("UPDATE questions SET correct_answer = ? WHERE id = ?", (new_answer, id))
    qa_conn.commit()

# --- เริ่ม Streamlit ---
st.set_page_config(page_title="ระบบเจ้าหน้าที่", page_icon="🔐", layout="centered")
st.title("🔐 ระบบจัดการสำหรับเจ้าหน้าที่")

# --- Session State ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# --- Login เท่านั้น ---
if not st.session_state.logged_in:
    st.subheader("กรุณาเข้าสู่ระบบ")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"ยินดีต้อนรับ {username}")
            st.rerun()
        else:
            st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
else:
    st.sidebar.success(f"👤 ผู้ใช้งาน: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    # --- แสดงตารางคำถาม + แก้ไข correct_answer ---
    data = get_all_questions()
    df = pd.DataFrame(data, columns=["id", "question", "answer", "timestamp", "correct_answer"])

    if df.empty:
        st.info("ยังไม่มีข้อมูลคำถาม")
    else:
        st.subheader("📋 ตารางคำถามจากผู้ใช้งาน")
        st.dataframe(df, use_container_width=True)

        st.markdown("---")
        st.subheader("✅ เพิ่มหรือแก้ไขคำตอบที่ถูกต้องของ Chatbot")

        selected_id = st.selectbox("เลือกลำดับข้อความที่ต้องการแก้ไข", df["id"])
        current_answer = df.loc[df["id"] == selected_id, "correct_answer"].values[0]
        new_answer = st.text_area("แก้ไขคำตอบที่ถูกต้อง", value=current_answer if current_answer else "")

        if st.button("บันทึก"):
            if new_answer != current_answer:
                update_correct_answer(selected_id, new_answer)
                st.success(f"บันทึกคำตอบที่ถูกต้องของ ID {selected_id} แล้ว")
                st.rerun()
            else:
                st.info("ไม่มีการเปลี่ยนแปลงข้อมูล")