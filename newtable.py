import sqlite3

conn = sqlite3.connect("questions.db")
c = conn.cursor()

# สร้างตารางชั่วคราวโดยไม่เอาคอลัมน์ rating
c.execute("""
CREATE TABLE feedback_temp (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message_id INTEGER,
    satisfaction TEXT,
    feedback_text TEXT,
    timestamp TEXT
)
""")

# คัดลอกข้อมูลจากตารางเก่าไปตารางใหม่ (ไม่เอา rating)
c.execute("""
INSERT INTO feedback_temp (id, user_message_id, satisfaction, feedback_text, timestamp)
SELECT id, user_message_id, satisfaction, feedback_text, timestamp
FROM feedback
""")

# ลบตารางเก่า
c.execute("DROP TABLE feedback")

# เปลี่ยนชื่อตารางใหม่เป็นชื่อเดิม
c.execute("ALTER TABLE feedback_temp RENAME TO feedback")

conn.commit()
conn.close()

print("✅ ลบคอลัมน์ 'rating' เรียบร้อยแล้ว")