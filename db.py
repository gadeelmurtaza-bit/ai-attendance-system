import sqlite3
import os

DB_PATH = "database/attendance.db"

def init_db():
    # Create folder if missing
    os.makedirs("database", exist_ok=True)

    # Create DB file if missing
    if not os.path.exists(DB_PATH):
        open(DB_PATH, "w").close()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        image_path TEXT NOT NULL
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        date TEXT,
        time TEXT,
        status TEXT,
        FOREIGN KEY(student_id) REFERENCES students(id)
    )''')

    conn.commit()
    conn.close()


def add_student(name, path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO students (name, image_path) VALUES (?, ?)", (name, path))
    conn.commit()
    conn.close()


def fetch_students():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM students")
    data = c.fetchall()
    conn.close()
    return data


def record_attendance(student_id, date, time, status):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)",
              (student_id, date, time, status))
    conn.commit()
    conn.close()


def get_attendance():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT students.name, attendance.date, attendance.time, attendance.status
                 FROM attendance JOIN students
                 ON attendance.student_id = students.id""")
    rows = c.fetchall()
    conn.close()
    return rows
