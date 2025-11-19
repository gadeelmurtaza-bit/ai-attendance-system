import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# ------------------------------
#     Folders for Cloud
# ------------------------------
STUDENT_FOLDER = "students"
LOG_FOLDER = "logs"
LOG_FILE = f"{LOG_FOLDER}/attendance.csv"

os.makedirs(STUDENT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# ------------------------------
#  Save Attendance
# ------------------------------
def mark_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[name, now]], columns=["Name", "Time"])
    df.to_csv(LOG_FILE, mode="a", header=False, index=False)

# ------------------------------
#  Register Student
# ------------------------------
def register_student(name, image):
    path = f"{STUDENT_FOLDER}/{name}.jpg"
    cv2.imwrite(path, image)
    return path

# ------------------------------
#  Face Recognition
# ------------------------------
def recognize_face(image):
    temp = "temp.jpg"
    cv2.imwrite(temp, image)

    try:
        res = DeepFace.find(img_path=temp, db_path=STUDENT_FOLDER, enforce_detection=False)
        if len(res[0]) > 0:
            identity = res[0].iloc[0]['identity']
            return os.path.splitext(os.path.basename(identity))[0]
    except:
        return None
    return None

# ------------------------------
#  Streamlit UI
# ------------------------------
st.title("📸 AI Attendance System (Cloud Version)")

menu = st.sidebar.radio("Menu", ["Home", "Register Student", "Take Attendance", "View Attendance"])

if menu == "Home":
    st.write("Welcome to the AI Attendance System!")

elif menu == "Register Student":
    st.header("Register New Student")
    name = st.text_input("Enter Student Name")
    photo = st.camera_input("Capture Student Photo")

    if name and photo:
        bytes_data = np.frombuffer(photo.read(), np.uint8)
        image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        register_student(name, image)
        st.success(f"{name} registered successfully!")

elif menu == "Take Attendance":
    st.header("Take Attendance")
    photo = st.camera_input("Capture Photo")

    if photo:
        bytes_data = np.frombuffer(photo.read(), np.uint8)
        image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)

        student = recognize_face(image)
        if student:
            mark_attendance(student)
            st.success(f"Attendance marked for **{student}**")
        else:
            st.error("Face not recognized!")

elif menu == "View Attendance":
    st.header("Attendance Logs")

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE, names=["Name", "Time"])
        st.dataframe(df)
    else:
        st.info("No attendance yet.")
