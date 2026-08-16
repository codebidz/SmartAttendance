import os
import cv2

# ============================
# Create Dataset Folder
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "DataSet")

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# ============================
# Student Details Input
# ============================
student_name = input("Enter Student Name: ").strip()

roll_no = input("Enter Roll Number: ").strip()

# ============================
# Create Student Folder
# Structure:
# DataSet/Name/RollNo/
# ============================
student_folder = os.path.join(DATASET_PATH, student_name)

roll_folder = os.path.join(student_folder, roll_no)

os.makedirs(roll_folder, exist_ok=True)

# ============================
# Open Camera
# ============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()

print("\nPress SPACE to capture image")
print("Press Q to quit\n")

img_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Display Instructions
    cv2.putText(
        frame,
        f"Name: {student_name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Roll No: {roll_no}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press SPACE to Capture",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1)

    # SPACE key
    if key == 32:

        img_name = f"{student_name}_{roll_no}_{img_count}.jpg"

        img_path = os.path.join(roll_folder, img_name)

        cv2.imwrite(img_path, frame)

        print(f"Saved: {img_path}")

        img_count += 1

    # Q key
    elif key == ord('q'):
        break

# ============================
# Cleanup
# ============================
cap.release()

cv2.destroyAllWindows()

print("\nDataset collection completed.")