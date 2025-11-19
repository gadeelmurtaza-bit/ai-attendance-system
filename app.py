import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import insightface
from insightface.app import FaceAnalysis

from db import init_db, add_student, fetch_students, record_attendance, get_attendance

st.set_page_config(page_title="AI Attendance System", layout="wide")

# Load DB
init_db()

# Load face model
model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
model.prepare(ctx_id=0)

PASSWORD = "admin123"

st.sidebar.title("Login")
password = st.sidebar.text_input("Enter admin password", type="password")

if password != PASSWORD:
    st.warning("Incorrect Password")
    st.stop()

menu = st.sidebar.radio("Menu", ["Register Student", "Attendance", "Dashboard"])


# -------------------------------------------
# Generate embedding
# -------------------------------------------
def get_embedding(img: Image.Image):
    np_img = np.array(img)
    faces = model.get(np_img)

    if len(faces) == 0:
        return None

    return faces[0].embedding


# -------------------------------------------
# Register student
# -------------------------------------------
if menu == "Register Student":
    st.title("Register New Student")

    name = st.text_input("Student Name")
    uploaded = st.file_uploader("Upload Student Photo", type=["jpg", "jpeg", "png"])

    if uploaded and name:
        img = Image.open(uploaded)
        embedding = get_embedding(img)

        if embedding is None:
            st.error("No face detected. Try another image.")
            st.stop()

        save_path = f"student_images/{name}.jpg"
        img.save(save_path)

        add_student(name, save_path, embedding.astype(np.float32))
        st.success(f"{name} registered successfully!")


# -------------------------------------------
# Mark Attendance
# -------------------------------------------
elif menu == "Attendance":
    st.title("Mark Attendance")

    capture = st.camera_input("Take a picture")

    if capture:
        img = Image.open(capture)
        embed = get_embedding(img)

        if embed is None:
            st.error("No face detected!")
            st.stop()

        students = fetch_students()

        best_match = None
        best_score = -1

        for (sid, name, path, emb_bytes) in students:
            stu_emb = np.frombuffer(emb_bytes, dtype=np.float32)
            score = cosine_similarity([embed], [stu_emb])[0][0]

            if score > best_score:
                best_score = score
                best_match = (sid, name)

        # Similarity threshold
        if best_score >= 0.45:
            sid, name = best_match
            now = datetime.now()
            record_attendance(sid, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), "Present")
            st.success(f"Attendance marked for **{name}** (match={best_score:.2f})")
        else:
            st.error("Face not recognized.")


# -------------------------------------------
# Dashboard
# -------------------------------------------
elif menu == "Dashboard":
    st.title("Attendance Dashboard")

    data = get_attendance()
    df = pd.DataFrame(data, columns=["Name", "Date", "Time", "Status"])

    st.dataframe(df)

    if not df.empty:
        st.subheader("Attendance Count")
        st.bar_chart(df["Name"].value_counts())
