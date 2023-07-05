import cv2
import numpy as np
import os

def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])

    dk = sorted(dist, key=lambda x: x[0])[:k]

    labels = np.array(dk)[:, -1]

    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

def capture_face_data(dataset_path, dir_name, file_name):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    skip = 0
    face_data = []

    # Create the directory if it doesn't exist
    dir_path = os.path.join(dataset_path, dir_name)
    os.makedirs(dir_path, exist_ok=True)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret == False:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(frame, 1.1, 5)
        faces = sorted(faces, key=lambda f: f[2] * f[3])

        face_section = None

        for face in faces[-1:]:
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            offset = 10
            face_section = frame[y - offset : y + h + offset, x - offset : x + w + offset]
            face_section = cv2.resize(face_section, (100, 100))

            face_data.append(face_section)
            print(len(face_data))

        cv2.imshow("Frame", frame)
        cv2.imshow("Face section", face_section)

        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord("q") or len(face_data) >= 100:
            break

    face_data = np.asarray(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))
    print(face_data.shape)

    if len(face_data) > 0:
        file_path = os.path.join(dir_path, file_name + ".npy")
        np.save(file_path, face_data)
        print("Data saved at", file_path)
    else:
        print("No face data captured. Data not saved.")

    cap.release()
    cv2.destroyAllWindows()

# Provide the dataset path and default directory name
dataset_path = "E:/Courses/24.Project 01 - Face Recognition/data/"
dir_name = "default"
file_name = input("Enter your name: ")

# Call the function to capture and save face data
capture_face_data(dataset_path, dir_name, file_name)
