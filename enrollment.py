# server/enrollment.py
import argparse
import cv2
import face_recognition
import numpy as np
from .db import init_db, add_student_with_embedding

def enroll(name: str, roll: str, cam_index: int = 0):
    init_db()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Could not open webcam. Try camera index 1 (use --cam 1).")
        return

    print("📷 Webcam opened. Position the student's face and press 'q' to capture.")
    embedding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam.")
            break

        cv2.imshow("Enroll - press q to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rgb = frame[:, :, ::-1]  # BGR -> RGB
            face_locations = face_recognition.face_locations(rgb)
            if not face_locations:
                print("⚠ No face found in this frame. Adjust pose/lighting and try again.")
                continue
            encodings = face_recognition.face_encodings(rgb, face_locations)
            if not encodings:
                print("⚠ Face detected but encoding failed. Try a clearer image.")
                continue
            embedding = encodings[0]
            print("✅ Face captured!")
            break

    cap.release()
    cv2.destroyAllWindows()

    if embedding is None:
        print("⚠ Enrollment aborted: No embedding captured.")
        return

    # store
    student = add_student_with_embedding(name, roll, embedding)
    print(f"🎉 Enrollment successful: {student.name} ({student.roll}), id={student.id}")

if _name_ == "_main_":
    parser = argparse.ArgumentParser(description="Enroll student: capture face and save embedding")
    parser.add_argument("--name", required=True)
    parser.add_argument("--roll", required=True)
    parser.add_argument("--cam", type=int, default=0, help="camera index (0 or 1 etc)")
    args = parser.parse_args()
    enroll(args.name, args.roll, cam_index=args.cam)