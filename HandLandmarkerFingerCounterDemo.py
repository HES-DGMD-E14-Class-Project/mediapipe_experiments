import mediapipe as mp
import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


class Mediapipe_BodyModule:
    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.results = None
        # load overlay images
        folderPath = "images"
        self.number_images = {}
        for image in os.listdir(folderPath):
            number = int(os.path.splitext(image)[0])
            image = cv2.imread(f"{folderPath}/{image}")
            self.number_images[number] = image

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        fingers_shown = self.count_fingers(hand_landmarks_list, handedness_list)
        print(f"Fingers shown: {fingers_shown}")
        if fingers_shown > 0:
            h, w, c = self.number_images[fingers_shown].shape
            annotated_image[0:h, 0:w] = self.number_images[fingers_shown]

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
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(
                annotated_image,
                f"{handedness[0].category_name}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated_image

    # Async Call-back
    def print_result(
        self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ):
        # print('hand landmarker result: {}'.format(result))
        self.results = result

    def count_fingers(self, hand_landmarks, handedness_list):
        """
        Count the number of fingers shown using the landmarks from MediaPipe Hand Landmarker.
        Args:
            hand_landmarks (List[NormalizedLandmark]): List of landmarks for a single hand.
        Returns:
            int: Number of fingers shown.
        """
        if len(hand_landmarks) < 1:  # MediaPipe hand model has 21 landmarks
            print(f"len(hand_landmarks): {len(hand_landmarks)}")
            return 0

        handedness = handedness_list[0][0].category_name

        tips = [
            4,
            8,
            12,
            16,
            20,
        ]  # Landmarks corresponding to tips of thumb, index, middle, ring, and pinky fingers.
        bases = [
            2,
            6,
            10,
            14,
            18,
        ]  # Landmarks corresponding to the base of each finger.

        finger_count = 0

        # Checking each finger:
        for i, (tip, base) in enumerate(zip(tips, bases)):
            if i == 0:  # thumb
                if handedness == "Right":
                    # If the x-coordinate of the thumb's tip is to the right of its base for the right hand
                    if hand_landmarks[0][tip].x > hand_landmarks[0][base].x:
                        finger_count += 1
                else:  # assuming left hand if not right
                    # If the x-coordinate of the thumb's tip is to the left of its base for the left hand
                    if hand_landmarks[0][tip].x < hand_landmarks[0][base].x:
                        finger_count += 1
            else:
                # If the y-coordinate of the tip is below that of the base, the finger is raised.
                if hand_landmarks[0][tip].y < hand_landmarks[0][base].y:
                    finger_count += 1

        return finger_count

    def main(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="tasks/hand_landmarker.task"),
            num_hands=1,
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result,
        )

        video = cv2.VideoCapture(0)

        timestamp = 0
        with HandLandmarker.create_from_options(options) as landmarker:
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
                        mp_image.numpy_view(), self.results
                    )
                    cv2.imshow("Show", annotated_image)
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
