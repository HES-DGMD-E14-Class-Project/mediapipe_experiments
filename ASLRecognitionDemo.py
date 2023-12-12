import time
import mediapipe as mp
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import (
    GCNConv,
    BatchNorm,
    global_max_pool,
    LayerNorm,
    MessagePassing,
)
import torch.nn.functional as F

ASL_SIGNS_MAPPING = {
    0: "Sign 0",
    1: "Sign 1",
    2: "Sign 2",
    3: "Sign 3",
    4: "Sign 4",
    5: "Sign 5",
    6: "Sign 6",
    7: "Sign 7",
    8: "Sign 8",
    9: "Sign 9",
    10: "Sign 10",
    11: "Sign 11",
    12: "Sign 12",
    13: "Sign 13",
    14: "Sign 14",
    15: "Sign 15",
    16: "Sign 16",
    17: "Sign 17",
    18: "Sign 18",
    19: "Sign 19",
}

# Landmarks specified in _extract_relevant_landmarks
# Pose landmarks from the waist up (excluding wrists, hands, and face - just the nose)
pose_landmarks_waist_up_no_face = [0, 9, 10, 11, 12, 13, 14]

# Minimal set of face landmarks for the outline of the face, eyes, and lips
# For the face oval, keep the transition points from sides of the face, corners,
# points where the face starts curving toward the chin, and some points for the chin.
# For eyes and lips, keep the corners and skip every other point.
face_landmarks_minimal = {
    # Reduced face oval (keeping side points, corners, and a few chin points)
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    # Right eye (skip every other point)
    33,
    133,
    246,
    161,
    159,
    158,
    157,
    173,
    # Left eye (skip every other point)
    263,
    362,
    466,
    388,
    386,
    385,
    384,
    398,
    # Reduced lips (keeping corners and midpoints)
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
}

relevant_landmarks = {
    *["pose-" + str(i) for i in pose_landmarks_waist_up_no_face],
    *["face-" + str(i) for i in face_landmarks_minimal],
    *["right_hand-" + str(i) for i in range(21)],
    *["left_hand-" + str(i) for i in range(21)],
}

MAX_LANDMARKS = len(relevant_landmarks)

# Define natural connections as class attributes
HAND_CONNECTIONS = frozenset(
    [
        # Left hand palm
        ("left_hand-0", "left_hand-1"),
        ("left_hand-0", "left_hand-5"),
        ("left_hand-9", "left_hand-13"),
        ("left_hand-13", "left_hand-17"),
        ("left_hand-5", "left_hand-9"),
        ("left_hand-0", "left_hand-17"),
        # Left hand thumb
        ("left_hand-1", "left_hand-2"),
        ("left_hand-2", "left_hand-3"),
        ("left_hand-3", "left_hand-4"),
        # Left hand index finger
        ("left_hand-5", "left_hand-6"),
        ("left_hand-6", "left_hand-7"),
        ("left_hand-7", "left_hand-8"),
        # Left hand middle finger
        ("left_hand-9", "left_hand-10"),
        ("left_hand-10", "left_hand-11"),
        ("left_hand-11", "left_hand-12"),
        # Left hand ring finger
        ("left_hand-13", "left_hand-14"),
        ("left_hand-14", "left_hand-15"),
        ("left_hand-15", "left_hand-16"),
        # Left hand pinky
        ("left_hand-17", "left_hand-18"),
        ("left_hand-18", "left_hand-19"),
        ("left_hand-19", "left_hand-20"),
        # Right hand palm
        ("right_hand-0", "right_hand-1"),
        ("right_hand-0", "right_hand-5"),
        ("right_hand-9", "right_hand-13"),
        ("right_hand-13", "right_hand-17"),
        ("right_hand-5", "right_hand-9"),
        ("right_hand-0", "right_hand-17"),
        # Right hand thumb
        ("right_hand-1", "right_hand-2"),
        ("right_hand-2", "right_hand-3"),
        ("right_hand-3", "right_hand-4"),
        # Right hand index finger
        ("right_hand-5", "right_hand-6"),
        ("right_hand-6", "right_hand-7"),
        ("right_hand-7", "right_hand-8"),
        # Right hand middle finger
        ("right_hand-9", "right_hand-10"),
        ("right_hand-10", "right_hand-11"),
        ("right_hand-11", "right_hand-12"),
        # Right hand ring finger
        ("right_hand-13", "right_hand-14"),
        ("right_hand-14", "right_hand-15"),
        ("right_hand-15", "right_hand-16"),
        # Right hand pinky
        ("right_hand-17", "right_hand-18"),
        ("right_hand-18", "right_hand-19"),
        ("right_hand-19", "right_hand-20"),
    ]
)

POSE_CONNECTIONS = frozenset(
    [
        ("pose-0", "pose-1"),
        ("pose-1", "pose-2"),
        ("pose-2", "pose-3"),
        ("pose-3", "pose-7"),
        ("pose-0", "pose-4"),
        ("pose-4", "pose-5"),
        ("pose-5", "pose-6"),
        ("pose-6", "pose-8"),
        ("pose-9", "pose-10"),
        ("pose-11", "pose-12"),
        ("pose-11", "pose-13"),
        ("pose-13", "pose-15"),
        ("pose-15", "pose-17"),
        ("pose-12", "pose-14"),
        ("pose-14", "pose-16"),
        ("pose-16", "pose-18"),
        ("pose-11", "pose-23"),
        ("pose-12", "pose-24"),
        ("pose-23", "pose-24"),
    ]
)

FACE_CONNECTIONS = frozenset(
    [
        # Connections for FACEMESH_LIPS using available landmarks
        ("face-61", "face-146"),
        ("face-146", "face-91"),
        ("face-91", "face-181"),
        ("face-181", "face-84"),
        ("face-84", "face-17"),
        ("face-17", "face-314"),
        ("face-314", "face-405"),
        ("face-405", "face-321"),
        ("face-321", "face-375"),
        ("face-375", "face-291"),
        ("face-78", "face-95"),
        ("face-95", "face-88"),
        ("face-88", "face-178"),
        ("face-178", "face-87"),
        ("face-87", "face-14"),
        ("face-14", "face-317"),
        ("face-317", "face-402"),
        ("face-402", "face-318"),
        ("face-318", "face-324"),
        ("face-324", "face-308"),
        # Connections for FACEMESH_LEFT_EYE using available landmarks
        ("face-263", "face-249"),
        ("face-388", "face-387"),
        ("face-387", "face-386"),
        ("face-386", "face-385"),
        ("face-385", "face-384"),
        ("face-384", "face-398"),
        # Connections for FACEMESH_LEFT_EYEBROW using available landmarks
        ("face-276", "face-283"),
        ("face-300", "face-293"),
        ("face-293", "face-334"),
        ("face-334", "face-296"),
        ("face-296", "face-336"),
        # Connections for FACEMESH_RIGHT_EYE using available landmarks
        ("face-33", "face-7"),
        ("face-246", "face-161"),
        ("face-161", "face-160"),
        ("face-160", "face-159"),
        ("face-159", "face-158"),
        ("face-158", "face-157"),
        ("face-157", "face-173"),
        # Connections for FACEMESH_RIGHT_EYEBROW using available landmarks
        ("face-46", "face-53"),
        ("face-70", "face-63"),
        ("face-63", "face-105"),
        ("face-105", "face-66"),
        ("face-66", "face-107"),
        # Connections for FACEMESH_FACE_OVAL using available landmarks
        ("face-10", "face-338"),
        ("face-338", "face-297"),
        ("face-297", "face-332"),
        ("face-332", "face-284"),
        ("face-284", "face-251"),
        ("face-251", "face-389"),
        ("face-389", "face-356"),
        ("face-356", "face-454"),
        ("face-454", "face-323"),
        ("face-323", "face-361"),
        ("face-361", "face-288"),
        ("face-288", "face-397"),
        ("face-397", "face-365"),
        ("face-365", "face-379"),
        ("face-379", "face-378"),
        ("face-378", "face-400"),
        ("face-400", "face-377"),
        ("face-377", "face-152"),
        ("face-152", "face-148"),
        ("face-148", "face-176"),
        ("face-176", "face-149"),
        ("face-149", "face-150"),
        ("face-150", "face-136"),
        ("face-136", "face-172"),
        ("face-172", "face-58"),
        ("face-58", "face-132"),
        ("face-132", "face-93"),
        ("face-93", "face-234"),
        ("face-234", "face-127"),
        ("face-127", "face-162"),
        ("face-162", "face-21"),
        ("face-21", "face-54"),
        ("face-54", "face-103"),
        ("face-103", "face-67"),
        ("face-67", "face-109"),
        ("face-109", "face-10"),
    ]
)


def convert_connections_to_edge_indices(connections):
    edge_indices = []
    for start, end in connections:
        start_idx = int(start.split("-")[1])
        end_idx = int(end.split("-")[1])
        edge_indices.append([start_idx, end_idx])
    return torch.tensor(edge_indices, dtype=torch.long).t().contiguous()


# Convert the connections to edge indices
hand_edge_indices = convert_connections_to_edge_indices(HAND_CONNECTIONS)
pose_edge_indices = convert_connections_to_edge_indices(POSE_CONNECTIONS)
face_edge_indices = convert_connections_to_edge_indices(FACE_CONNECTIONS)

# Combine the edge indices for all types
combined_edge_indices = torch.cat(
    [hand_edge_indices, pose_edge_indices, face_edge_indices], dim=1
)


class MinimalDummyASLClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MinimalDummyASLClassifier, self).__init__()
        self.num_classes = num_classes

    def forward(self, data):
        # Generate a single tensor of random log probabilities for one graph
        random_log_probs = torch.log(torch.rand((1, self.num_classes)))
        # Normalize to ensure they sum up to 1
        random_log_probs -= torch.logsumexp(random_log_probs, dim=1, keepdim=True)
        return random_log_probs


# Initialize the minimal dummy model
num_classes = 20  # Total number of classes
minimal_dummy_model = MinimalDummyASLClassifier(num_classes)


# PyTorch Geometric model
class ASLGraphClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(ASLGraphClassifier, self).__init__()
        self.conv1 = GCNConv(num_features, 512)
        self.bn1 = BatchNorm(512)
        self.conv2 = GCNConv(512, 1024)
        self.bn2 = BatchNorm(1024)
        self.ln1 = LayerNorm(1024)  # Layer normalization
        self.lin1 = torch.nn.Linear(1024, 512)
        self.ln2 = LayerNorm(512)  # Layer normalization
        self.lin2 = torch.nn.Linear(512, num_classes)

        self.dropout = torch.nn.Dropout(p=0.7)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.ln1(x)  # Apply layer normalization
        x = self.dropout(x)

        x = global_max_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = self.ln2(x)  # Apply layer normalization
        x = self.dropout(x)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class Mediapipe_BodyModule:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
        self.mp_pose = mp.solutions.pose.Pose()
        self.mp_hands = mp.solutions.hands.Hands()

        self.frame_buffer = []
        self.buffer_size = 40  # Number of frames for the model
        self.previous_frame_landmarks = None

        num_features = 6  # 2 (coordinates) + 2 (velocity) + 2 (acceleration)
        num_classes = 20  # Number of ASL Signs
        self.model = MinimalDummyASLClassifier(num_classes=num_classes)
        # self.model.load_state_dict(torch.load("path_to_model_weights.pth"))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.max_landmarks = MAX_LANDMARKS
        self.previous_frame_landmarks = np.zeros((self.max_landmarks, 3))
        self.previous_velocity = np.zeros((self.max_landmarks, 3))

        self.classify_sign = False

    def extract_landmarks(self, hand_landmarks, face_landmarks, pose_landmarks):
        """
        Extracts relevant landmarks from hand, face, and pose detections.

        Returns an array where missing landmarks are represented with placeholder values.
        """
        # Initialize a dictionary with placeholder values for each relevant landmark
        landmark_dict = {lm: [-1, -1, -1] for lm in relevant_landmarks}

        # Process pose landmarks
        if pose_landmarks and pose_landmarks.pose_landmarks:
            for idx, landmark in enumerate(pose_landmarks.pose_landmarks.landmark):
                lm_key = f"pose-{idx}"
                if lm_key in relevant_landmarks:
                    landmark_dict[lm_key] = [landmark.x, landmark.y, landmark.z]

        # Process face landmarks
        if face_landmarks and face_landmarks.multi_face_landmarks:
            for face_lms in face_landmarks.multi_face_landmarks:
                for idx, landmark in enumerate(face_lms.landmark):
                    lm_key = f"face-{idx}"
                    if lm_key in relevant_landmarks:
                        landmark_dict[lm_key] = [landmark.x, landmark.y, landmark.z]

        # Process hand landmarks
        if hand_landmarks and hand_landmarks.multi_hand_landmarks:
            for hand_idx, hand_lms in enumerate(hand_landmarks.multi_hand_landmarks):
                hand_type = (
                    "right_hand"
                    if hand_landmarks.multi_handedness[hand_idx].classification[0].label
                    == "Right"
                    else "left_hand"
                )
                for idx, landmark in enumerate(hand_lms.landmark):
                    lm_key = f"{hand_type}-{idx}"
                    if lm_key in relevant_landmarks:
                        landmark_dict[lm_key] = [landmark.x, landmark.y, landmark.z]

        # Convert the landmark dictionary to an array of values
        extracted_landmarks = list(landmark_dict.values())

        return np.array(extracted_landmarks)

    def update_previous_landmarks(self, current_frame_landmarks):
        """
        Updates the stored data for the previous frame.
        """
        # Reset previous landmarks and velocity
        self.previous_frame_landmarks.fill(0)
        self.previous_velocity.fill(0)

        # Update only the landmarks present in current_frame_landmarks
        for i, landmark in enumerate(current_frame_landmarks):
            if i < self.max_landmarks:
                self.previous_frame_landmarks[i] = landmark

    def normalize_landmarks(self, landmarks):
        """
        Normalize landmarks by subtracting the centroid.
        """
        centroid = np.mean(landmarks, axis=0)
        normalized_landmarks = landmarks - centroid
        return normalized_landmarks

    def calculate_velocity_acceleration(self, current_landmarks):
        """
        Calculate velocity and acceleration for landmarks, ensuring shape consistency.
        """
        # Find the maximum length between current and previous landmarks
        max_length = max(len(current_landmarks), len(self.previous_frame_landmarks))

        # Pad current_landmarks and previous_frame_landmarks to the same length
        current_landmarks_padded = np.pad(
            current_landmarks,
            ((0, max_length - len(current_landmarks)), (0, 0)),
            "constant",
        )
        previous_landmarks_padded = np.pad(
            self.previous_frame_landmarks,
            ((0, max_length - len(self.previous_frame_landmarks)), (0, 0)),
            "constant",
        )

        # Calculate velocity
        velocity = current_landmarks_padded - previous_landmarks_padded

        # Ensure velocity and previous_velocity are padded to the same length
        max_velocity_length = max(len(velocity), len(self.previous_velocity))
        velocity_padded = np.pad(
            velocity, ((0, max_velocity_length - len(velocity)), (0, 0)), "constant"
        )
        previous_velocity_padded = np.pad(
            self.previous_velocity,
            ((0, max_velocity_length - len(self.previous_velocity)), (0, 0)),
            "constant",
        )

        # Calculate acceleration
        acceleration = velocity_padded - previous_velocity_padded

        # Update previous landmarks and velocity
        self.update_previous_landmarks(current_landmarks_padded)
        self.previous_velocity = velocity_padded

        return velocity_padded, acceleration

    def construct_edge_index(self):
        edge_indices = []
        for connection_set in [HAND_CONNECTIONS, POSE_CONNECTIONS, FACE_CONNECTIONS]:
            for start, end in connection_set:
                # Extracting integer indices directly from tuples
                start_idx = int(start.split("-")[1])
                end_idx = int(end.split("-")[1])
                edge_indices.append([start_idx, end_idx])
        return torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    def create_graph_data(self, frame_features):
        x = torch.tensor(frame_features, dtype=torch.float)
        edge_index = self.construct_edge_index()
        return Data(x=x, edge_index=edge_index)

    def convert_buffer_to_model_input(self):
        # Flatten the buffer to create one large array of all landmarks
        all_features = np.vstack(self.frame_buffer)

        # Convert features to tensor
        x = torch.tensor(all_features, dtype=torch.float)

        # Construct edge indices for the graph
        edge_indices = self.construct_combined_edge_index()

        # Create graph data
        return Data(x=x, edge_index=edge_indices)

    def construct_combined_edge_index(self):
        edges = []
        max_landmarks = max(len(frame) for frame in self.frame_buffer)

        # Iterate through each frame
        for frame_index in range(len(self.frame_buffer)):
            # Add edges within and between frames
            for connection_set in [
                HAND_CONNECTIONS,
                POSE_CONNECTIONS,
                FACE_CONNECTIONS,
            ]:
                for connection in connection_set:
                    # Split each element of the tuple and convert to int
                    start_idx = (
                        int(connection[0].split("-")[1]) + max_landmarks * frame_index
                    )
                    end_idx = (
                        int(connection[1].split("-")[1]) + max_landmarks * frame_index
                    )

                    edges.append([start_idx, end_idx])

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame for landmarks
        hand_landmarks = self.mp_hands.process(frame_rgb)
        face_landmarks = self.mp_face_mesh.process(frame_rgb)
        pose_landmarks = self.mp_pose.process(frame_rgb)

        # Extract and normalize landmarks
        extracted_landmarks = self.extract_landmarks(
            hand_landmarks, face_landmarks, pose_landmarks
        )
        normalized_landmarks = self.normalize_landmarks(extracted_landmarks)

        # Calculate velocity and acceleration
        velocity, acceleration = self.calculate_velocity_acceleration(
            normalized_landmarks
        )

        # Update previous landmarks
        self.update_previous_landmarks(normalized_landmarks)

        # Pad arrays to ensure consistent shape
        max_length = max(len(normalized_landmarks), len(velocity), len(acceleration))
        normalized_landmarks = np.pad(
            normalized_landmarks,
            ((0, max_length - len(normalized_landmarks)), (0, 0)),
            "constant",
        )
        velocity = np.pad(
            velocity, ((0, max_length - len(velocity)), (0, 0)), "constant"
        )
        acceleration = np.pad(
            acceleration, ((0, max_length - len(acceleration)), (0, 0)), "constant"
        )

        # Combine features
        combined_features = np.hstack((normalized_landmarks, velocity, acceleration))

        # Add combined features to frame buffer
        self.frame_buffer.append(combined_features)

        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

        if len(self.frame_buffer) == self.buffer_size:
            # Construct graph and perform model inference
            model_input = self.convert_buffer_to_model_input()
            predicted_sign = self.perform_inference(model_input)

            # Display the result
            annotated_image = self.display_result(frame, predicted_sign)
            return annotated_image

        return self.draw_landmarks_on_image(
            frame, hand_landmarks, face_landmarks, pose_landmarks
        )

    def draw_landmarks_on_image(
        self, rgb_image, hand_landmarks, face_landmarks, pose_landmarks
    ):
        annotated_image = np.copy(rgb_image)
        if hand_landmarks.multi_hand_landmarks:
            for hand_lms in hand_landmarks.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image, hand_lms, mp.solutions.hands.HAND_CONNECTIONS
                )

        if face_landmarks.multi_face_landmarks:
            for face_lms in face_landmarks.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    face_lms,
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                )

        if pose_landmarks.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                pose_landmarks.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
            )

        return annotated_image

    def display_result(self, frame, asl_sign_label):
        """
        Draw the classification result on the frame.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)  # Text position
        font_scale = 1  # Font scale
        color = (255, 0, 0)  # Color (B, G, R)
        thickness = 2  # Thickness

        # Put the text on the frame
        cv2.putText(
            frame, asl_sign_label, org, font, font_scale, color, thickness, cv2.LINE_AA
        )
        return frame

    def perform_inference(self, model_input):
        # If using the real model, uncomment the line below
        # self.model.eval()  # Set the model to evaluation mode if needed

        # Perform inference
        output = self.model(model_input)
        print(f"After inference, output is {output}")

        # Find the index of the class with the highest probability
        predicted_class_index = output.argmax(dim=1).item()
        print(f"> predicted_class_index {predicted_class_index}")

        # Retrieve the corresponding sign string
        predicted_sign = ASL_SIGNS_MAPPING.get(predicted_class_index, "Unknown")
        print(f"> predicted_sign {predicted_sign}")
        return predicted_sign

    def main(self):
        video = cv2.VideoCapture(0)
        start_capture = False
        classification_done = False
        capture_duration = 5  # Duration to capture frames for classification
        predicted_sign = "Waiting..."  # Default text
        countdown_active = False
        countdown_start_time = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("Ignoring empty frame")
                break

            # Process frame for landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_landmarks = self.mp_hands.process(frame_rgb)
            face_landmarks = self.mp_face_mesh.process(frame_rgb)
            pose_landmarks = self.mp_pose.process(frame_rgb)

            # Draw landmarks on the frame
            annotated_image = self.draw_landmarks_on_image(
                frame, hand_landmarks, face_landmarks, pose_landmarks
            )

            if countdown_active:
                elapsed_time = time.time() - countdown_start_time
                countdown = capture_duration - int(elapsed_time)
                if countdown > 0:
                    cv2.putText(
                        annotated_image,
                        str(countdown),
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                else:
                    countdown_active = False
                    start_capture = True
                    capture_start_time = time.time()
                    self.frame_buffer = []  # Clear the buffer

            elif classification_done:
                annotated_image = self.display_result(annotated_image, predicted_sign)
            elif not start_capture and not countdown_active:
                # Display "Waiting..." text only when not capturing and countdown is inactive
                annotated_image = self.display_result(annotated_image, "Waiting...")

            # Check for 'S' key press to start countdown
            if (
                cv2.waitKey(5) & 0xFF == ord("s")
                and not start_capture
                and not countdown_active
            ):
                print("Ready to classify next sign...")
                classification_done = False
                countdown_active = True
                countdown_start_time = time.time()

            # Process and classify the frame if capture is started
            if start_capture and not classification_done:
                # Extract and normalize landmarks
                extracted_landmarks = self.extract_landmarks(
                    hand_landmarks, face_landmarks, pose_landmarks
                )
                normalized_landmarks = self.normalize_landmarks(extracted_landmarks)
                # Calculate velocity and acceleration
                velocity, acceleration = self.calculate_velocity_acceleration(
                    normalized_landmarks
                )

                # Combine features
                combined_features = np.hstack(
                    (normalized_landmarks, velocity, acceleration)
                )
                self.frame_buffer.append(combined_features)

                # Update capturing text with frame count
                capture_text = (
                    f"Capturing... {len(self.frame_buffer)}/{self.buffer_size}"
                )
                cv2.putText(
                    annotated_image,
                    capture_text,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Check if the capture duration has passed
                if len(self.frame_buffer) >= self.buffer_size:
                    predicted_sign = self.perform_inference(
                        self.convert_buffer_to_model_input()
                    )
                    print(f"Classified as: {predicted_sign}")
                    classification_done = True
                    start_capture = False

            # Show the annotated frame
            cv2.imshow("Show", annotated_image)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    body_module = Mediapipe_BodyModule()
    body_module.main()
