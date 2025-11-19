import streamlit as st
import os
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace

from db import init_db, add_student, fetch_students, record_attendance, get_attendance

st.set_page_config(page_title="AI Attendance System", layout="wide")

# Initialize DB
init_db()

PASSWORD = "admin123"

# Sidebar Login
st.sidebar.title("🔐 Admin Login")
password = st.sidebar.text_input("Enter Password", type="password")

if password != PASSWORD:
    st.warning("Enter correct admin password")
    st.stop()

menu = st.sidebar.radio("Menu", ["Register Student", "Attendance", "Dashboard"])
st.sidebar.success("Logged in Successfully!")

# -------------------------------------------
# Face Verification using DeepFace
# -------------------------------------------
def verify_face(img1_path, img2_path):
    try:
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            enforce_detection=False
        )
        return result["verified"]
    except:
        return False


# -------------------------------------------
# Register Student
# -------------------------------------------
if menu == "Register Student":
    st.title("🧑‍🎓 Register New Student")

    name = st.text_input("Student Name")
    uploaded_img = st.file_uploader("Upload Student Photo", type=["jpg", "png", "jpeg"])

    if uploaded_img and name:
        os.makedirs("student_images", exist_ok=True)

        save_path = f"student_images/{name}.jpg"
        with open(save_path, "wb") as f:
            f.write(uploaded_img.getbuffer())

        add_student(name, save_path)
        st.success(f"Student '{name}' registered!")


# -------------------------------------------
# Attendance
# -------------------------------------------
elif menu == "Attendance":
    st.title("📷 Mark Attendance")

    students = fetch_students()

    captured = st.camera_input("Take a picture to mark attendance")

    if captured:
        img = Image.open(captured)
        os.makedirs("temp", exist_ok=True)

        user_img_path = "temp/user.jpg"
        img.save(user_img_path)

        marked = False

        for stu in students:
            student_id, student_name, stored_path = stu

            if verify_face(user_img_path, stored_path):
                now = datetime.now()
                record_attendance(
                    student_id,
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    "Present"
                )

                st.success(f"Attendance marked for **{student_name}**")
                marked = True
                break

        if not marked:
            st.error("Face not recognized!")


# -------------------------------------------
# Dashboard
# -------------------------------------------
elif menu == "Dashboard":
    st.title("📊 Attendance Dashboard")

    data = get_attendance()
    df = pd.DataFrame(data, columns=["Name", "Date", "Time", "Status"])

    st.dataframe(df)

    st.subheader("Attendance Count")
    if not df.empty:
        st.bar_chart(df["Name"].value_counts())
