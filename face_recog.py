from deepface import DeepFace

def verify_face(img1, img2):
    try:
        result = DeepFace.verify(img1_path=img1, img2_path=img2, enforce_detection=False)
        return result["verified"]
    except:
        return False
