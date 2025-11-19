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
menu = ["Start Webcam", "View Attendance", "Upload Student Images"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Upload Student Images":
    uploaded_files = st.file_uploader("Upload Student Images", accept_multiple_files=True, type=['jpg','jpeg','png'])
    if uploaded_files:
        for file in uploaded_files:
            image = Image.open(file)
            image.save(f'students_images/{file.name}')
        st.success("Images uploaded successfully! Refresh the app to load new students.")

elif choice == "Start Webcam":
    stframe = st.empty()
    run = st.checkbox('Start Webcam')
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to access webcam")
            break

        imgS = cv2.resize(frame, (0,0), None, 0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(known_encodings, encodeFace)
            faceDis = face_recognition.face_distance(known_encodings, encodeFace)
            if len(faceDis) > 0:
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = student_names[matchIndex]
                    mark_attendance(name)
                    y1,x2,y2,x1 = faceLoc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, name, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()

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
