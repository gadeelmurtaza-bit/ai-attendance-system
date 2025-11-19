import sqlite3
import numpy as np
import pickle

DB_PATH = "students.db"

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            photo_path TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

# Add a student
def add_student(name, photo_path, embedding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    emb_blob = pickle.dumps(embedding)
    c.execute("INSERT INTO students (name, photo_path, embedding) VALUES (?, ?, ?)",
              (name, photo_path, emb_blob))
    conn.commit()
    conn.close()

# Get all students
def get_all_students():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, photo_path, embedding FROM students")
    rows = c.fetchall()
    conn.close()
    students = []
    for name, photo_path, emb_blob in rows:
        students.append({
            "name": name,
            "photo_path": photo_path,
            "embedding": pickle.loads(emb_blob)
        })
    return students
