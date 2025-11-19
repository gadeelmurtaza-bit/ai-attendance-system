import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
import face_recognition

from db import init_db, add_student, fetch_students, record_attendance, get_attendance

st.set_page_config(page_title="AI Attendance PRO", layout="wide")

init_db()

PASSWORD = "admin123"

# ----------------------------------------------------
# SIDEBAR MENU
# ----------------------------------------------------
st.sidebar.title("🔐 Admin Login")
password = st.sidebar.text_input("Enter Password", type="password")

if password != PASSWORD:
    st.warning("Enter correct admin password to access system")
    st.stop()

st.sidebar.success("Logged in as Admin")
menu = st.sidebar.radio("Menu", ["Register Student", "Attendance", "Dashboard"])

# ----------------------------------------------------
# REGISTER STUDENT
# ----------------------------------------------------
if menu == "Register Student":
    st.title("🧑‍🎓 Register New Student")

    name = st.text_input("Student Name")
    img_file = st.file_uploader("Upload Student Photo", type=["jpg", "jpeg", "png"])

    if img_file and name:
        image_path = f"student_images/{name}.jpg"
        with open(image_path, "wb") as f:
            f.write(img_file.getbuffer())

        add_student(name, image_path)
        st.success(f"Student '{name}' registered!")

# ----------------------------------------------------
# ATTENDANCE MARKING
# ----------------------------------------------------
elif menu == "Attendance":
    st.title("📷 Mark Attendance")

    students = fetch_students()
    known_encodings = []
    known_names = []

    # Load encodings
    for stu in students:
        img = face_recognition.load_image_file(stu[2])
        enc = face_recognition.face_encodings(img)

        if enc:
            known_encodings.append(enc[0])
            known_names.append(stu)

    camera = st.camera_input("Take picture")

    if camera:
        img = camera.getvalue()
        nparr = np.frombuffer(img, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_encodings(rgb)

        for face in faces:
            matches = face_recognition.compare_faces(known_encodings, face)
            dist = face_recognition.face_distance(known_encodings, face)

            idx = np.argmin(dist)

            if matches[idx]:
                stu_id = known_names[idx][0]
                stu_name = known_names[idx][1]

                now = datetime.now()
                record_attendance(stu_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M"), "Present")

                st.success(f"Attendance Marked: {stu_name}")
            else:
                st.error("Unknown face detected!")

# ----------------------------------------------------
# DASHBOARD
# ----------------------------------------------------
elif menu == "Dashboard":
    st.title("📊 Attendance Dashboard")

    data = get_attendance()
    df = pd.DataFrame(data, columns=["Name", "Date", "Time", "Status"])
    st.dataframe(df)

    st.subheader("Attendance Count")
    st.bar_chart(df["Name"].value_counts())
