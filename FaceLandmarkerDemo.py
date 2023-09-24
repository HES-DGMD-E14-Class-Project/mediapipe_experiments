import mediapipe as mp
import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


class Mediapipe_BodyModule:
    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.results = None

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return annotated_image

    # Create a face landmarker instance with the live stream mode:
    def print_result(
        self, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ):
        print("face landmarker result: {}".format(result))
        self.results = result

    def main(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="tasks/face_landmarker.task"),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result,
        )

        video = cv2.VideoCapture(0)

        timestamp = 0
        with FaceLandmarker.create_from_options(options) as landmarker:
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
                        mp_image.numpy_view(), self.results
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
