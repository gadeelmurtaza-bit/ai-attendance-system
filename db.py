import sqlite3

def init_db():
    conn = sqlite3.connect("database/attendance.db")
    c = conn.cursor()

    # Student Table
    c.execute('''CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        image_path TEXT NOT NULL
    )''')

    # Attendance Table
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

def add_student(name, image_path):
    conn = sqlite3.connect("database/attendance.db")
    c = conn.cursor()
    c.execute("INSERT INTO students (name, image_path) VALUES (?, ?)", (name, image_path))
    conn.commit()
    conn.close()

def fetch_students():
    conn = sqlite3.connect("database/attendance.db")
    c = conn.cursor()
    c.execute("SELECT * FROM students")
    data = c.fetchall()
    conn.close()
    return data

def record_attendance(student_id, date, time, status):
    conn = sqlite3.connect("database/attendance.db")
    c = conn.cursor()
    c.execute("INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)",
              (student_id, date, time, status))
    conn.commit()
    conn.close()

def get_attendance():
    conn = sqlite3.connect("database/attendance.db")
    c = conn.cursor()
    c.execute("""SELECT students.name, attendance.date, attendance.time, attendance.status
                 FROM attendance JOIN students
                 ON attendance.student_id = students.id""")
    data = c.fetchall()
    conn.close()
    return data
