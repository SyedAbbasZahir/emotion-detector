# import cv2
# from deepface import DeepFace

# # Load face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Start webcam
# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#     cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     raise IOError("Cannot open webcam")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     # Convert to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     # Analyze only if a face exists
#     try:
#         result = DeepFace.analyze(frame,
#                                   actions=['emotion'],
#                                   enforce_detection=False)   # Important fix!
#         emotion = result[0]["dominant_emotion"]
#     except:
#         emotion = "No Face"

#     # Draw bounding boxes & emotion
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
#         cv2.putText(frame, emotion, (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1, (0, 255, 0), 2)

#     # Show window
#     cv2.imshow("Emotion Detector", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace

st.title("Emotion Detector")

img_file = st.camera_input("Take a picture")

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    result = DeepFace.analyze(frame,
                              actions=['emotion'],
                              enforce_detection=False)

    emotion = result[0]["dominant_emotion"]

    st.image(frame, channels="BGR")
    st.write("Detected Emotion:", emotion)