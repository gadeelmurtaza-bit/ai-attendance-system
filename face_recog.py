import face_recognition
import cv2
import numpy as np

def encode_face(image_path):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(rgb)

    if enc:
        return enc[0]
    return None
