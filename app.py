import streamlit as st
import pandas as pd
import os
from PIL import Image
import numpy as np
from io import BytesIO
import zipfile

# DB functions
from db import add_student, init_db

# InsightFace
import insightface
from insightface.app import FaceAnalysis

# Initialize DB
init_db()

# Load InsightFace model
st.text("Loading Face Recognition Model...")
model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
model.prepare(ctx_id=0)
st.success("Model loaded!")

st.title("Bulk Student Registration")

# -------------------------------------------
# Function to get embedding from an image
# -------------------------------------------
def get_embedding(img: Image.Image):
    np_img = np.array(img)
    faces = model.get(np_img)

    if len(faces) == 0:
        return None

    return faces[0].embedding

# -------------------------------------------
# Upload CSV/Excel and ZIP of photos
# -------------------------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
photo_zip = st.file_uploader("Upload Zip of Photos", type=["zip"])

if uploaded_file and photo_zip:
    # Extract ZIP
    with zipfile.ZipFile(BytesIO(photo_zip.read())) as zip_ref:
        zip_ref.extractall("student_images/")

    st.success("Photos extracted successfully!")

    # Read CSV/Excel
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Register each student
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
                    st.warning(f"No face detected for {name} ({roll})")
            else:
                st.error(f"Photo not found: {photo_name}")
        except Exception as e:
            st.error(f"Error processing row {idx}: {e}")

    st.success("Bulk registration completed!")
