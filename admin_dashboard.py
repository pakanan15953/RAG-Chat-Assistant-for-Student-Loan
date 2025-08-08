import sqlite3
import streamlit as st
import pandas as pd
import hashlib

DB_PATH = "questions.db"
USER_DB_PATH = "adminMN.db"  # ไฟล์แยกสำหรับข้อมูลผู้ใช้

# ฟังก์ชันสำหรับจัดการฐานข้อมูล users
def init_user_database():
    """สร้างตาราง users ในไฟล์ password_user.db หากยังไม่มี และเพิ่มผู้ใช้เริ่มต้น"""
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    
    # สร้างตาราง users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'staff',
            full_name TEXT,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    ''')
    
    # เพิ่มผู้ใช้เริ่มต้นหากยังไม่มี
    default_users = [
        ("admin", "password", "admin", "ผู้ดูแลระบบ", "admin@company.com"),
        ("manager", "secret123", "manager", "ผู้จัดการ", "manager@company.com"), 
        ("staff", "secret", "staff", "เจ้าหน้าที่", "staff@company.com")
    ]
    
    for username, password, role, full_name, email in default_users:
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
        if cursor.fetchone()[0] == 0:
            password_hash = hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, full_name, email) VALUES (?, ?, ?, ?, ?)",
                (username, password_hash, role, full_name, email)
            )
    
    conn.commit()
    conn.close()

def get_user_credentials():
    """ดึงข้อมูล login ทั้งหมดจากฐานข้อมูล adminMN.db"""
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username, password_hash FROM users WHERE is_active = 1")
    credentials = dict(cursor.fetchall())
    conn.close()
    return credentials

def get_user_info(username):
    """ดึงข้อมูลผู้ใช้ทั้งหมด"""
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND is_active = 1", (username,))
    user_info = cursor.fetchone()
    conn.close()
    return user_info

def update_last_login(username):
    """อัพเดทเวลาล็อกอินล่าสุด"""
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?",
        (username,)
    )
    conn.commit()
    conn.close()

def get_all_users():
    """ดึงข้อมูลผู้ใช้ทั้งหมดสำหรับการจัดการ"""
    conn = sqlite3.connect(USER_DB_PATH)
    df = pd.read_sql_query("""
        SELECT id, username, role, full_name, email, created_at, last_login, is_active
        FROM users 
        ORDER BY created_at DESC
    """, conn)
    conn.close()
    return df

def hash_password(password):
    """เข้ารหัสรหัสผ่านด้วย SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username, password):
    """ตรวจสอบ username และ password จากฐานข้อมูล"""
    credentials = get_user_credentials()
    hashed_password = hash_password(password)
    
    if username in credentials and credentials[username] == hashed_password:
        update_last_login(username)
        return True
    return False

def login_form():
    """แสดงฟอร์มล็อกอิน"""
    st.title("🔐 เข้าสู่ระบบผู้ดูแล")
    st.write("กรุณาเข้าสู่ระบบเพื่อเข้าถึง Admin Dashboard")
    
    with st.form("login_form"):
        username = st.text_input("👤 ชื่อผู้ใช้")
        password = st.text_input("🔑 รหัสผ่าน", type="password")
        submit_button = st.form_submit_button("เข้าสู่ระบบ")
        
        if submit_button:
            if check_login(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("✅ เข้าสู่ระบบสำเร็จ!")
                st.rerun()
            else:
                st.error("❌ ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
    
    # แสดงข้อมูลการทดสอบ
    with st.expander("ℹ️ ข้อมูลสำหรับทดสอบ"):
        st.write("**บัญชีผู้ใช้สำหรับทดสอบ:**")
        st.code("""
Username: admin     | Password: password
Username: manager   | Password: secret123  
Username: staff     | Password: secret
        """)

def logout():
    """ออกจากระบบ"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

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

def main_dashboard():
    """หน้าหลักของแดชบอร์ด"""
    # ดึงข้อมูลผู้ใช้
    user_info = get_user_info(st.session_state.username)
    
    # Header พร้อมปุ่มออกจากระบบ
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🧑‍💼 Admin Dashboard")
        st.write("แสดงคำถามที่ถูกถาม พร้อมแก้ไขคำตอบที่ถูกต้องได้")
    with col2:
        if user_info:
            st.write(f"👋 สวัสดี **{user_info[4]}**")  # full_name
            st.caption(f"สถานะ: {user_info[3]}")  # role
        else:
            st.write(f"👋 สวัสดี **{st.session_state.username}**")
        
        if st.button("🚪 ออกจากระบบ"):
            logout()
    
    st.divider()
    
    # แสดงข้อมูลไฟล์ฐานข้อมูล
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📊 **ฐานข้อมูลคำถาม**: `{DB_PATH}`")
    with col2:
        st.info(f"👥 **ฐานข้อมูลผู้ใช้**: `{USER_DB_PATH}`")
    
    # โหลดและแสดงข้อมูล
    df = load_data()
    
    if df.empty:
        st.info("ยังไม่มีข้อมูลคำถามในระบบ")
    else:
        # แสดงสถิติ
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 จำนวนคำถามทั้งหมด", len(df))
        with col2:
            answered = df['correct_answer'].notna().sum()
            st.metric("✅ มีคำตอบที่ถูกต้องแล้ว", answered)
        with col3:
            unanswered = len(df) - answered
            st.metric("❓ ยังไม่มีคำตอบที่ถูกต้อง", unanswered)
        with col4:
            # นับจำนวนผู้ใช้ทั้งหมด
            total_users = len(get_all_users())
            st.metric("👥 จำนวนผู้ใช้", total_users)
        
        st.divider()
        
        # แสดงตารางแบบ editable เฉพาะคอลัมน์ correct_answer
        st.subheader("📝 แก้ไขคำตอบที่ถูกต้อง")
        editable_df = df[["id", "question", "answer", "correct_answer", "timestamp"]].copy()
        
        edited_df = st.data_editor(
            editable_df,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "question": st.column_config.TextColumn("คำถาม", width="large"),
                "answer": st.column_config.TextColumn("คำตอบที่ AI ตอบ", width="large"),
                "correct_answer": st.column_config.TextColumn("คำตอบที่ถูกต้อง (แก้ไขได้)", width="large"),
                "timestamp": st.column_config.DatetimeColumn("เวลาที่ถาม", width="medium")
            },
            disabled=["id", "question", "answer", "timestamp"],
            use_container_width=True,
            num_rows="dynamic",
            height=400
        )
        
        # ปุ่มบันทึก
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("💾 บันทึกทั้งหมด", type="primary"):
                changes = 0
                for i in range(len(df)):
                    old = df.loc[i, "correct_answer"] if pd.notna(df.loc[i, "correct_answer"]) else ""
                    new = edited_df.loc[i, "correct_answer"] if pd.notna(edited_df.loc[i, "correct_answer"]) else ""
                    if old != new:
                        update_correct_answer(df.loc[i, "id"], new if new else None)
                        changes += 1
                
                if changes > 0:
                    st.success(f"✅ บันทึกคำตอบที่ถูกต้องแล้ว {changes} รายการ")
                    st.rerun()
                else:
                    st.info("ไม่มีการเปลี่ยนแปลงข้อมูล")
    
    # ส่วนจัดการผู้ใช้ (สำหรับ admin เท่านั้น)
    if user_info and user_info[3] == 'admin':  # role = admin
        st.divider()
        with st.expander("👥 จัดการผู้ใช้ระบบ", expanded=False):
            users_df = get_all_users()
            st.dataframe(
                users_df,
                column_config={
                    "id": "ID",
                    "username": "ชื่อผู้ใช้",
                    "role": "สถานะ",
                    "full_name": "ชื่อเต็ม",
                    "email": "อีเมล",
                    "created_at": "วันที่สร้าง",
                    "last_login": "ล็อกอินล่าสุด",
                    "is_active": "สถานะการใช้งาน"
                },
                use_container_width=True
            )
    
    # ส่วนลบข้อมูล
    st.divider()
    with st.expander("🗑️ ลบข้อมูลทั้งหมด", expanded=False):
        st.warning("⚠️ **คำเตือน:** การลบข้อมูลไม่สามารถย้อนกลับได้!")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🗑️ ลบข้อมูลคำถาม", type="secondary"):
                delete_all_data()
                st.success("✅ ลบข้อมูลคำถามทั้งหมดเรียบร้อยแล้ว")
                st.rerun()


# Main App
def main():
    st.set_page_config(
        page_title="Admin Dashboard - คำถามทั้งหมด", 
        page_icon="📊", 
        layout="wide"
    )
    # ตรวจสอบสถานะการเข้าสู่ระบบ
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None

    init_user_database()

    if st.session_state.authenticated:
        main_dashboard()
    else:
        login_form()

if __name__ == "__main__":
    main()