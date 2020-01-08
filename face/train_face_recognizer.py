from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import argparse
import pickle
from imutils import paths
import imutils
import os
import cv2
import face_recognition
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='directory of input images')
ap.add_argument('-o', '--recognizer', required=True,
                help='face recognizer output recognizer path')
ap.add_argument('-m', '--model', required=False, default='knn',
                help='face recognizer model type')
args = vars(ap.parse_args())
recognizer_path = args['recognizer']
arg_model = args['model']


print('[INFO] loading image dataset ...')
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


print('[INFO] training model ...')
le = LabelEncoder()
dataset_labels = le.fit_transform(dataset_names)

if arg_model == 'svc':
    model = SVC(C=1.0, kernel='linear', probability=True)
    model.fit(dataset_embeddings, dataset_labels)
elif arg_model == 'knn':
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(dataset_embeddings, dataset_labels)
elif arg_model == 'rf':
    model = RandomForestClassifier()
    model.fit(dataset_embeddings, dataset_labels)


print('[INFO] saving model ...')
recognizer = {}
recognizer['names'] = le.classes_
recognizer['model'] = model
with open(recognizer_path, 'wb') as f:
    f.write(pickle.dumps(recognizer))
