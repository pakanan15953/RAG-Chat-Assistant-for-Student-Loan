import sqlite3
import bcrypt

# --- เชื่อมต่อฐานข้อมูล ---
user_conn = sqlite3.connect('secure_users.db')
user_c = user_conn.cursor()

# --- สร้าง table users ถ้ายังไม่มี ---
user_c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password BLOB NOT NULL,
    role TEXT DEFAULT 'user'
)
''')
user_conn.commit()

# --- ฟังก์ชันเพิ่มผู้ใช้ ---
def add_user(username, plain_password, role='user'):
    hashed_pw = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())
    try:
        user_c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hashed_pw, role)
        )
        user_conn.commit()
        print(f"เพิ่มผู้ใช้เรียบร้อย: {username} ({role})")
    except sqlite3.IntegrityError:
        print(f"ผู้ใช้ {username} มีอยู่แล้ว")

# --- ตัวอย่างการเพิ่มผู้ใช้ ---
# เพิ่ม admin
add_user("pakanan", "tantiwut", role="admin")

# เพิ่ม user ปกติ
# add_user("user1", "password", role="user")

# --- ปิดการเชื่อมต่อ ---
user_conn.close()