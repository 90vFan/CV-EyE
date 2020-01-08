import face_recognition
import cv2
import numpy as np
import argparse
import os
from imutils import paths
import imutils
import PIL
import pickle
import re


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=False,
                help='directory of image dataset with human faces')
ap.add_argument('-i', '--image', required=False,
                help='input image for face recognization')
ap.add_argument('-v', '--video', required=False,
                help='input video file, or camera interface (default: 0) for recognization')
ap.add_argument('-r', '--recognizer', required=False,
                help='model to recognize face')
args = vars(ap.parse_args())
arg_dataset = args['dataset']
arg_image = args['image']
arg_video = args['video']
arg_recognizer = args['recognizer']


# ====================================
# Load image dataset
# ====================================
if arg_dataset:
    dataset_names = []
    dataset_embeddings = []
    image_paths = list(paths.list_images(arg_dataset))
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

if arg_recognizer:
    with open(arg_recognizer, 'rb') as f:
        recognizer = pickle.loads(f.read())
    recognizer_names = recognizer['names']
    recognizer_model = recognizer['model']


# ======================================
# predict face name
# ======================================
def predict_face_name_by_compare(dataset_names, dataset_embeddings, face_encodings):
    pred_face_names = []
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

        pred_face_names.append(name)

    return pred_face_names


def predict_face_name_by_model(recognizer_names, recognizer_model, face_encodings):
    pred_face_names = []
    if not face_encodings:
        return pred_face_names

    predict = recognizer_model.predict_proba(face_encodings)
    # print(predict)
    for pred in predict:
        pred_name = 'Unknown'
        max_idx = np.argmax(pred)
        if pred[max_idx] > 0.3:
            pred_name = recognizer_names[max_idx]
        pred_face_names.append(pred_name)

    return pred_face_names


def draw_face_rect(frame, face_locations, pred_face_names):
    # display rectangle of face
    for (top, right, bottom, left), name in zip(face_locations, pred_face_names):
        # top *= 4
        # right *= 4
        # bottom *= 4
        # left *= 4

        # draw a box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom + 16), font, 0.45, (255, 255, 0), 1)
        # print('Detect face: ' + name)


def video_recognizer():
    arg_video = args['video']
    if re.search(r'^\d*$', args['video']):
        arg_video = int(arg_video)

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

        if arg_dataset:
            pred_face_names = predict_face_name_by_compare(dataset_names, dataset_embeddings, face_encodings)
        elif arg_recognizer:
            pred_face_names = predict_face_name_by_model(
                recognizer_names, recognizer_model, face_encodings)

        draw_face_rect(frame, face_locations, pred_face_names)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def image_recognizer():
    image = cv2.imread(arg_image)
    rgb_image = image[:, : , ::-1]

    face_locations = face_recognition.face_locations(rgb_image, model='cnn')
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if arg_dataset:
        pred_face_names = predict_face_name_by_compare(dataset_names, dataset_embeddings, face_encodings)
    elif arg_recognizer:
        pred_face_names = predict_face_name_by_model(
            recognizer_names, recognizer_model, face_encodings)

    draw_face_rect(image, face_locations, pred_face_names)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if arg_image:
        image_recognizer()
    elif arg_video:
        video_recognizer()
