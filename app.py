import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime
from PIL import Image

st.set_page_config(page_title="AI Attendance System", layout="wide")
st.title("📸 AI Attendance System")

# --- Auto-create folders/files ---
if not os.path.exists('students_images'):
    os.makedirs('students_images')
if not os.path.exists('attendance.csv'):
    pd.DataFrame(columns=['Name', 'Date', 'Time']).to_csv('attendance.csv', index=False)

# --- Load student images ---
images = []
student_names = []
for file in os.listdir('students_images'):
    curImg = cv2.imread(f'students_images/{file}')
    images.append(curImg)
    student_names.append(os.path.splitext(file)[0].upper())

# --- Encode faces ---
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encode_list.append(encodes[0])
    return encode_list

known_encodings = find_encodings(images)

# --- Attendance marking ---
def mark_attendance(name):
    df = pd.read_csv('attendance.csv')
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    if not ((df['Name'] == name) & (df['Date'] == date_str)).any():
        df = pd.concat([df, pd.DataFrame([[name, date_str, time_str]], columns=['Name', 'Date', 'Time'])])
        df.to_csv('attendance.csv', index=False)

# --- Sidebar Menu ---
menu = ["Upload Student Images", "Mark Attendance", "View Attendance"]
choice = st.sidebar.selectbox("Menu", menu)

# --- Upload student images ---
if choice == "Upload Student Images":
    uploaded_files = st.file_uploader("Upload Student Images", accept_multiple_files=True, type=['jpg','jpeg','png'])
    if uploaded_files:
        for file in uploaded_files:
            image = Image.open(file)
            image.save(f'students_images/{file.name}')
        st.success("Images uploaded successfully! Refresh the app to load new students.")

# --- Mark attendance from uploaded photo ---
elif choice == "Mark Attendance":
    uploaded_file = st.file_uploader("Upload a student photo to mark attendance", type=['jpg','jpeg','png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        facesCurFrame = face_recognition.face_locations(frame)
        encodesCurFrame = face_recognition.face_encodings(frame, facesCurFrame)

        if len(encodesCurFrame) == 0:
            st.warning("No face detected in the uploaded image!")
        else:
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(known_encodings, encodeFace)
                faceDis = face_recognition.face_distance(known_encodings, encodeFace)
                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        name = student_names[matchIndex]
                        mark_attendance(name)
                        st.success(f"✅ Attendance marked for {name}")
                    else:
                        st.error("❌ Face not recognized!")
            # Optional: show uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

# --- View attendance ---
elif choice == "View Attendance":
    st.subheader("Attendance Records")
    df = pd.read_csv('attendance.csv')
    if df.empty:
        st.info("No attendance records yet.")
    else:
        st.dataframe(df)
        all_students = [os.path.splitext(f)[0].upper() for f in os.listdir('students_images')]
        today = datetime.now().strftime('%Y-%m-%d')
        present_today = df[df['Date'] == today]['Name'].tolist()
        absent_today = [s for s in all_students if s not in present_today]

        st.success(f"Present Today: {', '.join(present_today) if present_today else 'None'}")
        st.warning(f"Absent Today: {', '.join(absent_today) if absent_today else 'None'}")
