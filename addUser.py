import sqlite3
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

conn = sqlite3.connect("adminMN.db")
cursor = conn.cursor()
cursor.execute("""
    INSERT INTO users (username, password_hash, role, full_name, email) 
    VALUES (?, ?, ?, ?, ?)
""", ("thi", hash_password("fasai"), "staff", "ผู้ใช้ใหม่", "fasai.com")) 
#ถ้าจะเพิ่มID-PASSWORD แก้บรรทัด12 user คือ ไอดี password คือรหัส staff คือ โรลผู้ใช้ สุดท้ายคือ email
conn.commit()
conn.close()

