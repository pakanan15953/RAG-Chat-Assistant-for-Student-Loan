import sqlite3

# เชื่อมต่อกับฐานข้อมูล (หากยังไม่มีไฟล์ฐานข้อมูลจะถูกสร้างขึ้น)
conn = sqlite3.connect('adminMN.db')  # หรือใส่ :memory: เพื่อสร้างฐานข้อมูลชั่วคราวใน RAM

# สร้าง cursor สำหรับรันคำสั่ง SQL
cursor = conn.cursor()

# คำสั่ง SQL สำหรับสร้างตาราง users
create_table_sql = '''
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
);
'''

# รันคำสั่ง SQL
cursor.execute(create_table_sql)

# บันทึกการเปลี่ยนแปลงและปิดการเชื่อมต่อ
conn.commit()
conn.close()
