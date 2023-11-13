import mediapipe as mp
import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import load_model


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
GESTURES = [
    "okay",
    "peace",
    "thumbs up",
    "thumbs down",
    "call me",
    "stop",
    "rock",
    "live long",
    "fist",
    "smile",
]


class Mediapipe_BodyModule:
    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.results = None
        # gesture classification model
        self.model = load_model("models/mp_hand_gesture")

    def draw_landmarks_on_image(self, rgb_image, detection_result, frame):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)
        x, y, c = frame.shape

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style(),
            )

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            left = int(min(x_coordinates) * width)
            top = int(min(y_coordinates) * height) - MARGIN
            right = int(max(x_coordinates) * width)
            bottom = int(max(y_coordinates) * height) + MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(
                annotated_image,
                f"{handedness[0].category_name}",
                (left, top),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

            landmarks = [[int(lm.x * x), int(lm.y * y)] for lm in hand_landmarks]

            # Gesture Classification
            prediction = self.model.predict([landmarks])
            gesture = GESTURES[np.argmax(prediction)]
            print(f"gesture ==> {gesture}")
            cv2.putText(
                annotated_image,
                gesture,
                (left, bottom),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated_image

    # Create a hand landmarker instance with the live stream mode:
    def print_result(
        self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ):
        # print("hand landmarker result: {}".format(result))
        self.results = result

    def main(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="tasks/hand_landmarker.task"),
            num_hands=2,
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result,
        )

        video = cv2.VideoCapture(0)

        timestamp = 0
        with HandLandmarker.create_from_options(options) as landmarker:
            # The landmarker is initialized. Use it here.
            # ...
            while video.isOpened():
                # Capture frame-by-frame
                ret, frame = video.read()

                if not ret:
                    print("Ignoring empty frame")
                    break

                timestamp += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                landmarker.detect_async(mp_image, timestamp)

                if not (self.results is None):
                    annotated_image = self.draw_landmarks_on_image(
                        mp_image.numpy_view(), self.results, frame
                    )
                    cv2.imshow("Show", annotated_image)
                    print("showing detected image")
                else:
                    cv2.imshow("Show", frame)

                if cv2.waitKey(5) & 0xFF == ord("q"):
                    print("Closing Camera Stream")
                    break

            video.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    body_module = Mediapipe_BodyModule()
    body_module.main()
