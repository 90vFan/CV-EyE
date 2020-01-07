import face_recognition
import cv2
import numpy as np
import argparse
import os
from imutils import paths
import imutils
import PIL


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--dataset', required=False, help='directory of input images')
ap.add_argument('-v', '--video', required=False, default=0)
args = vars(ap.parse_args())
arg_dataset = args['dataset']
arg_video = args['video']


# ====================================
# Load image dataset
# ====================================
dataset_names = []
dataset_embeddings = []
image_paths = list(paths.list_images(args["dataset"]))
for image_path in image_paths:
    name = image_path.split(os.path.sep)[-2]
    # image = face_recognition.load_image_file(image_path)
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)

    face_locations = face_recognition.face_locations(image, model='cnn')
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for face_encoding in face_encodings:
        dataset_names.append(name)
        dataset_embeddings.append(np.array(face_encoding))

cap = cv2.VideoCapture(arg_video)

while True:
    ret, frame = cap.read()

    small_frame = frame
    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # BGR -> RGB
    # rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(dataset_embeddings, face_encoding)
        name = 'Unknown'

        # face_distances = face_recognition.face_distance(dataset_embeddings, face_encoding)
        # best_match_idx = np.argmin(face_distances)
        # if matches[best_match_idx]:
        #     name = dataset_names[best_match_idx]

        # if True in matches:
        #     first_match_idx = matches.index(True)
        #     name = dataset_names[first_match_idx]

        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matchedIdxs:
            name = dataset_names[i]
            counts[name] = counts.get(name, 0) + 1

        if counts:
            name = max(counts, key=counts.get)

        face_names.append(name)

    # display rectangle of face
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # top *= 4
        # right *= 4
        # bottom *= 4
        # left *= 4

        # draw a box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom + 20), font, 0.45, (255, 255, 0), 1)
        # print('Detect face: ' + name)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
