import streamlit as st
import sqlite3
import os
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
from deepface import DeepFace

from db import init_db, add_student, fetch_students, record_attendance, get_attendance

st.set_page_config(page_title="AI Attendance PRO", layout="wide")
init_db()

PASSWORD = "admin123"

# Sidebar login
st.sidebar.title("🔐 Admin Login")
password = st.sidebar.text_input("Enter Password", type="password")

if password != PASSWORD:
    st.warning("Enter correct admin password to continue")
    st.stop()

st.sidebar.success("Logged in Successfully!")
menu = st.sidebar.radio("Menu", ["Register Student", "Attendance", "Dashboard"])

# ------------------------------------------
# Verify Faces (DeepFace)
# ------------------------------------------
def verify_face(img1, img2):
    try:
        result = DeepFace.verify(img1_path=img1, img2_path=img2, enforce_detection=False)
        return result["verified"]
    except:
        return False

# ------------------------------------------
# Register Student
# ------------------------------------------
if menu == "Register Student":
    st.title("🧑‍🎓 Register New Student")

    name = st.text_input("Student Name")
    img = st.file_uploader("Upload Student Photo", type=["jpg", "jpeg", "png"])

    if img and name:
        path = f"student_images/{name}.jpg"
        with open(path, "wb") as f:
            f.write(img.getbuffer())

        add_student(name, path)
        st.success(f"Student '{name}' Registered!")

# ------------------------------------------
# Attendance
# ------------------------------------------
elif menu == "Attendance":
    st.title("📷 Take Picture to Mark Attendance")

    students = fetch_students()

    cam = st.camera_input("Take a picture")

    if cam:
        img_bytes = cam.getvalue()
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        user_img_path = "temp_user.jpg"
        cv2.imwrite(user_img_path, frame)

        matched = False

        for stu in students:
            student_id, name, db_img_path = stu

            if verify_face(user_img_path, db_img_path):
                now = datetime.now()
                record_attendance(student_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M"), "Present")
                st.success(f"Attendance Marked for {name}")
                matched = True
                break

        if not matched:
            st.error("Face Not Recognized!")

# ------------------------------------------
# Dashboard
# ------------------------------------------
elif menu == "Dashboard":
    st.title("📊 Attendance Dashboard")

    data = get_attendance()
    df = pd.DataFrame(data, columns=["Name", "Date", "Time", "Status"])

    st.dataframe(df)

    st.subheader("Attendance Count")
    st.bar_chart(df["Name"].value_counts())
