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

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip = 0
dataset_path = 'E:/Courses/24.Project 01 - Face Recognition/data/default'
face_data = []
labels = []

class_id = 0
names = {}

for fx in os.listdir(dataset_path):
    if fx.endswith(".npy"):
        names[class_id] = fx[:-4]
        print("loaded" + fx)
        data_item = np.load(os.path.join(dataset_path, fx))

        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

# Load the sunglass image with an alpha channel
sunglass_image = cv2.imread("sunglass.png", cv2.IMREAD_UNCHANGED)

# Extract the sunglass region (alpha channel) and the corresponding RGB channels
sunglass_alpha = sunglass_image[:, :, 3] / 255.0
sunglass_rgb = sunglass_image[:, :, :3]

while True:
    ret, frame = cap.read()
    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(frame, 1.1, 5)

    for face in faces:
        x, y, w, h = face

        offset = 10
        face_section = frame[y - offset: y + h + offset, x - offset: x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())

        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Resize the sunglass image to fit the face width
        resized_sunglass_rgb = cv2.resize(sunglass_rgb, (w, h))

        # Compute the region of interest (ROI) for the face and sunglass
        roi_frame = frame[y:y + h, x:x + w]
        roi_sunglass_rgb = resized_sunglass_rgb.copy()
        roi_sunglass_alpha = cv2.resize(sunglass_alpha, (w, h))

        # Overlay the sunglass on the face region
        for c in range(3):
            roi_frame[:, :, c] = (roi_sunglass_rgb[:, :, c] * roi_sunglass_alpha) + \
                                 (roi_frame[:, :, c] * (1 - roi_sunglass_alpha))

    cv2.imshow("faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
