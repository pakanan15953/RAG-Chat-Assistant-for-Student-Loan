import streamlit as st
import sqlite3
import pandas as pd
import bcrypt
from datetime import datetime

# --- เชื่อมต่อฐานข้อมูลผู้ใช้ ---
user_conn = sqlite3.connect('secure_users.db', check_same_thread=False)
user_c = user_conn.cursor()

# --- สร้าง table users ถ้ายังไม่มี ---
user_c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password BLOB NOT NULL
)
''')
user_conn.commit()

# --- ตรวจสอบและเพิ่มคอลัมน์ role ถ้ายังไม่มี ---
user_c.execute("PRAGMA table_info(users)")
columns = [col[1] for col in user_c.fetchall()]
if 'role' not in columns:
    user_c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
    user_conn.commit()
    print("เพิ่มคอลัมน์ 'role' เรียบร้อยแล้ว")

# --- ฟังก์ชันเพิ่มผู้ใช้ ---
def add_user(username, plain_password, role='user'):
    hashed_pw = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())
    try:
        user_c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hashed_pw, role)
        )
        user_conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# --- ฟังก์ชันตรวจสอบผู้ใช้ ---
def verify_user(username, password):
    user_c.execute("SELECT password, role FROM users WHERE username = ?", (username,))
    result = user_c.fetchone()
    if result:
        hashed_pw_db, role = result
        # ตรวจสอบชนิดข้อมูลก่อน verify
        if isinstance(hashed_pw_db, str):
            hashed_pw = hashed_pw_db.encode('utf-8')
        else:
            hashed_pw = hashed_pw_db  # เป็น bytes อยู่แล้ว
        if bcrypt.checkpw(password.encode('utf-8'), hashed_pw):
            return True, role
    return False, None

# --- เชื่อมต่อฐานข้อมูลคำถาม ---
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
    qa_c.execute("SELECT id, question, answer, timestamp, correct_answer FROM questions")
    return qa_c.fetchall()

# --- Streamlit Page Config ---
st.set_page_config(page_title="ระบบเจ้าหน้าที่", page_icon="🔐", layout="wide")
st.title("🔐 ระบบจัดการสำหรับเจ้าหน้าที่")

# --- Session State ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'role' not in st.session_state:
    st.session_state.role = "user"

# --- Login ---
if not st.session_state.logged_in:
    st.subheader("กรุณาเข้าสู่ระบบ")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        valid, role = verify_user(username, password)
        if valid:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = role
            st.success(f"ยินดีต้อนรับ {username} ({role})")
            st.rerun()
        else:
            st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")

# --- Logged in ---
else:
    st.sidebar.success(f"👤 ผู้ใช้งาน: {st.session_state.username} ({st.session_state.role})")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = "user"
        st.rerun()

    # --- ดึงข้อมูลคำถาม ---
    data = get_all_questions()
    df = pd.DataFrame(data, columns=["id", "question", "answer", "timestamp", "correct_answer"])

    st.subheader("📋 ตารางคำถามทั้งหมด")
    if df.empty:
        st.info("ยังไม่มีข้อมูลคำถาม")
    else:
        st.dataframe(df, use_container_width=True, height=600)

    # --- ฟีเจอร์สำหรับ Admin ---
    if st.session_state.role == "admin":
        st.markdown("---")
        st.subheader("➕ เพิ่มคำถามใหม่")
        new_question = st.text_area("คำถาม")
        new_answer = st.text_area("คำตอบ")
        if st.button("เพิ่มคำถาม"):
            if new_question and new_answer:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                qa_c.execute(
                    "INSERT INTO questions (question, answer, timestamp) VALUES (?, ?, ?)",
                    (new_question, new_answer, timestamp)
                )
                qa_conn.commit()
                st.success("เพิ่มคำถามเรียบร้อยแล้ว")
                st.rerun()
            else:
                st.error("กรุณากรอกคำถามและคำตอบก่อน")

        st.markdown("---")
        st.subheader("✏️ แก้ไขคำถาม/คำตอบ/คำตอบที่ถูกต้อง")
        selected_id = st.selectbox("เลือกลำดับข้อความที่ต้องการแก้ไข", df["id"])
        current_question = df.loc[df["id"] == selected_id, "question"].values[0]
        current_answer = df.loc[df["id"] == selected_id, "answer"].values[0]
        current_correct = df.loc[df["id"] == selected_id, "correct_answer"].values[0]

        edit_question = st.text_area("คำถาม", value=current_question)
        edit_answer = st.text_area("คำตอบ", value=current_answer)
        edit_correct = st.text_area("คำตอบที่ถูกต้อง", value=current_correct if current_correct else "")

        if st.button("บันทึกการแก้ไข"):
            qa_c.execute(
                "UPDATE questions SET question=?, answer=?, correct_answer=? WHERE id=?",
                (edit_question, edit_answer, edit_correct, selected_id)
            )
            qa_conn.commit()
            st.success(f"บันทึกการแก้ไข ID {selected_id} เรียบร้อยแล้ว")
            st.rerun()

        st.markdown("---")
        st.subheader("🗑️ ลบคำถาม")
        delete_id = st.selectbox("เลือกลำดับข้อความที่จะลบ", df["id"], key="delete_id")
        if st.button("ลบคำถาม"):
            qa_c.execute("DELETE FROM questions WHERE id=?", (delete_id,))
            qa_conn.commit()
            st.success(f"ลบคำถาม ID {delete_id} เรียบร้อยแล้ว")
            st.rerun()
