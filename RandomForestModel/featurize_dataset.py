import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = 'data/'

data = []
labels = []

    

def landmarks_helper(x_list, y_list, features, landmark_results):
    # Helper function filter out relevant features out of an image/landmark and then add it to the appropriate list
    for hand_landmarks in landmark_results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_list.append(x)
            y_list.append(y)
            
            # features.append(x - min(x_list))
            # features.append(y - min(y_list))
            
        # IMPORTANT: We are forced to use an extra for loop here because we need to have created x_list and y_list already
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            features.append(x - min(x_list))
            features.append(y - min(y_list))

    data.append(features)
    labels.append(curr_directory)


for curr_directory in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, curr_directory)):
        features = []

        # Used to store the landmarks
        x_list, y_list = [], []

        img = cv2.imread(os.path.join(DATA_DIR, curr_directory, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            landmarks_helper(x_list, y_list, features, results)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
