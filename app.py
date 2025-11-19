import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from io import BytesIO
import zipfile

from insightface.app import FaceAnalysis
from db import add_student, init_db, get_all_students  # Your DB functions

# For camera input
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -----------------------------------------
# Initialize Database
# -----------------------------------------
init_db()

# -----------------------------------------
# Load Face Recognition Model
# -----------------------------------------
st.text("Loading Face Recognition Model...")
model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
model.prepare(ctx_id=0)
st.success("Face Recognition Model Loaded!")

# -----------------------------------------
# Ensure folder exists
# -----------------------------------------
def ensure_folder(folder_path):
    if os.path.exists(folder_path):
        if not os.path.isdir(folder_path):
            os.remove(folder_path)
            os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)

ensure_folder("student_images")

# -----------------------------------------
# Background
# -----------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;color:#0077cc;'>Smart Student Registration & Attendance System</h1>
    <p style='text-align:center;'>Upload students, register new ones, and take attendance using AI-powered Face Recognition.</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------
# Get Face Embedding
# -----------------------------------------
def get_embedding(img: Image.Image):
    np_img = np.array(img)
    faces = model.get(np_img)
    if len(faces) == 0:
        return None
    return faces[0].embedding

# -----------------------------------------
# Sidebar Navigation
# -----------------------------------------
menu = st.sidebar.radio("Menu", ["Bulk Registration", "Add Student", "Take Attendance"])

# =========================================
# 1️⃣ BULK REGISTRATION
# =========================================
if menu == "Bulk Registration":
    st.header("📂 Bulk Student Registration")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    photo_zip = st.file_uploader("Upload ZIP of Photos", type=["zip"])

    if uploaded_file and photo_zip:
        ensure_folder("student_images")

        # Extract ZIP
        with zipfile.ZipFile(BytesIO(photo_zip.read())) as zip_ref:
            zip_ref.extractall("student_images/")
        st.success("Photos extracted successfully!")

        # Read CSV/Excel
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.info("Starting registration...")

        for idx, row in df.iterrows():
            try:
                name = str(row["Name"])
                roll = str(row["Roll Number"])
                photo_name = str(row["Photo Filename"])
                img_path = f"student_images/{photo_name}"

                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    embedding = get_embedding(img)

                    if embedding is not None:
                        add_student(f"{name} ({roll})", img_path, embedding.astype(np.float32))
                        st.success(f"Registered: {name} ({roll})")
                    else:
                        st.warning(f"No face detected for: {name} ({roll})")
                else:
                    st.error(f"Photo not found: {photo_name}")
            except Exception as e:
                st.error(f"Error processing row {idx}: {e}")

        st.success("Bulk registration completed!")

# =========================================
# 2️⃣ MANUAL STUDENT REGISTRATION
# =========================================
elif menu == "Add Student":
    st.header("➕ Register New Student")
    name = st.text_input("Student Name")
    roll = st.text_input("Roll Number")
    photo = st.file_uploader("Upload Student Photo (JPG)", type=["jpg", "jpeg"])

    if st.button("Register Student"):
        if not name or not roll or not photo:
            st.error("Please fill all fields and upload a photo.")
        else:
            img = Image.open(photo)
            embedding = get_embedding(img)

            if embedding is None:
                st.error("No face detected! Try another photo.")
            else:
                ensure_folder("student_images")
                save_path = f"student_images/{roll}.jpg"
                img.save(save_path)
                add_student(f"{name} ({roll})", save_path, embedding.astype(np.float32))
                st.success(f"Student Registered: {name} ({roll})")

# =========================================
# 3️⃣ ATTENDANCE
# =========================================
elif menu == "Take Attendance":
    st.header("📸 Take Attendance")

    # Load students from DB
    db_students = get_all_students()
    attendance_df = pd.DataFrame({
        "Name": [s["name"] for s in db_students],
        "Status": ["Absent"] * len(db_students)
    })
    st.subheader("📋 Attendance Table")
    st.dataframe(attendance_df)

    # Face Matching Function
    def match_face(live_embedding, db_students, threshold=0.6):
        for student in db_students:
            db_emb = student["embedding"]
            dist = np.linalg.norm(live_embedding - db_emb)
            if dist < threshold:
                return student["name"]
        return None

    # Video Transformer
    class FaceAttendance(VideoTransformerBase):
        def __init__(self):
            self.marked_students = set()

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            faces = model.get(img)
            if len(faces) > 0:
                embedding = faces[0].embedding
                matched_name = match_face(embedding, db_students)
                if matched_name:
                    self.marked_students.add(matched_name)
            return frame

    ctx = webrtc_streamer(key="attendance", video_transformer_factory=FaceAttendance)

    if ctx.video_transformer:
        marked = ctx.video_transformer.marked_students
        if marked:
            attendance_df.loc[attendance_df["Name"].isin(marked), "Status"] = "Present"
            st.dataframe(attendance_df)
