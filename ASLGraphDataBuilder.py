import os
import json
import numpy as np
import pandas as pd
import logging

from tqdm import tqdm
from scipy.signal import savgol_filter
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    filename="asl_graph_builder.log", format="%(asctime)s %(message)s", filemode="w"
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ASLGraphDataBuilder:
    # Constants for pose landmarks
    POSE_NOSE = 0
    POSE_LEFT_SHOULDER = 11
    POSE_RIGHT_SHOULDER = 12
    POSE_LEFT_ELBOW = 13
    POSE_RIGHT_ELBOW = 14
    HAND_LEFT_WRIST = 0
    HAND_RIGHT_WRIST = 0

    def __init__(self, base_dir, signs_to_process, max_files_per_sign, target_frames):
        """
        Initializes the ASLGraphDataBuilder class.

        :param base_dir: The base directory where the dataset is stored.
        :param signs_to_process: A list of signs that should be processed. If None, all signs will be processed.
        :param max_files_per_sign: The maximum number of files to process for each sign.
        :param target_frames: The target number of frames to interpolate to for each example.
        """
        self.base_dir = base_dir
        self.signs_to_process = signs_to_process
        self.max_files_per_sign = max_files_per_sign
        self.target_frames = target_frames

        # Landmarks specified in _extract_relevant_landmarks
        # Pose landmarks from the waist up (excluding wrists, hands, and face - just the nose)
        self.pose_landmarks_waist_up_no_face = [0, 9, 10, 11, 12, 13, 14]

        # Minimal set of face landmarks for the outline of the face, eyes, and lips
        # For the face oval, keep the transition points from sides of the face, corners,
        # points where the face starts curving toward the chin, and some points for the chin.
        # For eyes and lips, keep the corners and skip every other point.
        self.face_landmarks_minimal = {
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

        self.relevant_landmarks = {
            *["pose-" + str(i) for i in self.pose_landmarks_waist_up_no_face],
            *["face-" + str(i) for i in self.face_landmarks_minimal],
            *["right_hand-" + str(i) for i in range(21)],
            *["left_hand-" + str(i) for i in range(21)],
        }

        self.expected_landmark_count = len(self.relevant_landmarks)

        # Load the train dataframe and label map
        self.train_df = pd.read_csv(os.path.join(self.base_dir, "train.csv"))

        if not signs_to_process:
            self.signs_to_process = self.train_df["sign"].unique().tolist()
        else:
            self.signs_to_process = signs_to_process

        with open(
            os.path.join(self.base_dir, "sign_to_prediction_index_map.json")
        ) as f:
            self.label_map = json.load(f)

    def _filter_files_by_sign(self, sign):
        """
        Filters and returns the file paths for a given sign.

        :param sign: The sign to filter files for.
        :return: A list of file paths for the given sign.
        """
        sign_files = self.train_df[self.train_df["sign"] == sign]["path"].tolist()[
            : self.max_files_per_sign
        ]
        return [os.path.join(self.base_dir, f) for f in sign_files]

    def _remove_empty_frames(self, df):
        """
        Removes frames from the dataframe where all landmarks are missing.

        :param df: The dataframe to remove empty frames from.
        :return: The dataframe with empty frames removed.
        """
        df = df.groupby("frame").filter(
            lambda group: group[["x", "y", "z"]].notna().any().any()
        )
        return df

    def _extract_relevant_landmarks(self, df):
        """
        Filters the dataframe to only include a minimal set of relevant landmarks for facial contours.

        :param df: The dataframe to filter.
        :return: The dataframe with only relevant landmarks.
        """
        # Filter the DataFrame
        df["landmark_type"] = df["row_id"].apply(lambda x: x.split("-")[1])
        df["landmark_index"] = df["row_id"].apply(lambda x: int(x.split("-")[2]))
        df["landmark_id"] = df["landmark_type"] + "-" + df["landmark_index"].astype(str)

        # Include 'arms_configuration' in the filtered DataFrame
        if "arms_configuration" in df.columns:
            frames_configuration = df[["frame", "arms_configuration"]].drop_duplicates()
            df_filtered = df[df["landmark_id"].isin(self.relevant_landmarks)]
            df_filtered = df_filtered[
                [
                    "landmark_id",
                    "row_id",
                    "landmark_index",
                    "x",
                    "y",
                    "z",
                    "frame",
                    "type",
                ]
            ]
            # Merge the arms_configuration back into the filtered DataFrame
            df_filtered = df_filtered.merge(
                frames_configuration, on="frame", how="left"
            )
        else:
            df_filtered = df[df["landmark_id"].isin(self.relevant_landmarks)]
            df_filtered = df_filtered[
                [
                    "landmark_id",
                    "row_id",
                    "landmark_index",
                    "x",
                    "y",
                    "z",
                    "frame",
                    "type",
                ]
            ]

        # # Debugging: Checking available landmarks after filtering
        # for frame in df_filtered['frame'].unique():
        #     frame_data = df_filtered[df_filtered['frame'] == frame]
        #     filtered_landmarks = frame_data['landmark_id'].unique()
        #     print(f"Frame {frame}: Remaining landmarks after filtering: {filtered_landmarks}")

        return df_filtered

    def _handle_nan_values(self, df):
        """
        Handles NaN values in the dataframe by interpolating missing values and then dropping any remaining NaN values.

        :param df: The dataframe to handle NaN values in.
        :return: The dataframe with NaN values handled.
        """
        df = self._interpolate_landmarks(df)
        df.dropna(subset=["x", "y", "z"], inplace=True)
        return df

    def _interpolate_landmarks(self, df):
        """
        Fills missing values for each landmark type in the dataframe by carrying forward the last known value.

        :param df: The dataframe to fill values in.
        :return: The dataframe with filled values.
        """
        for landmark_type in df["type"].unique():
            mask = df["type"] == landmark_type

            # Carry forward the last known value
            df.loc[mask, "x"] = df.loc[mask, "x"].ffill().bfill()
            df.loc[mask, "y"] = df.loc[mask, "y"].ffill().bfill()
            df.loc[mask, "z"] = df.loc[mask, "z"].ffill().bfill()

        return df

    def _drop_z_coordinate(self, df):
        """
        Drops the z-coordinate from the dataframe if it exists.

        :param df: The dataframe to drop the z-coordinate from.
        :return: The dataframe without the z-coordinate.
        """
        return df.drop(columns=["z"], errors="ignore")

    def _normalize_coordinates(self, df):
        """
        Normalizes the x and y coordinates in the dataframe by subtracting the centroid.

        :param df: The dataframe to normalize coordinates in.
        :return: The dataframe with normalized coordinates.
        """
        centroid = df[["x", "y"]].mean().tolist()
        df["x"] = df["x"] - centroid[0]
        df["y"] = df["y"] - centroid[1]
        return df

    def _smooth_landmarks(self, df, window_length, polyorder):
        """
        Applies a Savitzky-Golay filter to smooth the landmark coordinates in the dataframe.

        :param df: The dataframe to smooth landmarks in.
        :param window_length: The length of the filter window.
        :param polyorder: The order of the polynomial used to fit the samples.
        :return: The dataframe with smoothed landmarks.
        """
        if window_length % 2 == 0 or window_length <= polyorder:
            raise ValueError("window_length must be an odd number and >= polyorder.")

        for landmark_type in df["type"].unique():
            mask = df["type"] == landmark_type

            if window_length > mask.sum():
                raise ValueError(
                    "window_length is too large for the number of frames. Reduce window_length or use more frames."
                )

            df.loc[mask, "x"] = savgol_filter(
                df.loc[mask, "x"], window_length, polyorder
            )
            df.loc[mask, "y"] = savgol_filter(
                df.loc[mask, "y"], window_length, polyorder
            )

        return df

    def _interpolate_frames(self, df):
        """
        Interpolates or reduces the number of frames in the dataframe to match the target number of frames.

        :param df: The dataframe to interpolate or reduce frames in.
        :return: The dataframe with interpolated or reduced frames.
        """
        num_frames = len(df["frame"].unique())
        df["frame"] = df["frame"].rank(method="dense").astype(int) - 1
        df.sort_values(by=["frame", "landmark_index"], inplace=True)

        if num_frames < self.target_frames:
            return self._increase_frames(df)
        elif num_frames > self.target_frames:
            return self._reduce_frames(df)

        return df

    def _increase_frames(self, df):
        current_frames = df["frame"].nunique()

        if current_frames >= self.target_frames:
            return df  # No need to increase frames if already sufficient

        # Calculate the number of interpolation steps needed
        steps_needed = np.ceil(
            (self.target_frames - current_frames) / (current_frames - 1)
        ).astype(int)

        interpolated_rows = []
        new_frame_number = (
            current_frames  # Start numbering new frames after existing frames
        )

        for frame_number in range(current_frames - 1):
            frame_data_start = df[df["frame"] == frame_number]
            frame_data_end = df[df["frame"] == frame_number + 1]

            # Check if frame data lengths match and contain all landmarks
            if len(frame_data_start) != len(frame_data_end):
                raise ValueError(
                    f"Mismatch in landmark counts between frames {frame_number} and {frame_number + 1}"
                )

            for step in range(1, steps_needed + 1):
                t = step / steps_needed
                for landmark_index in range(len(frame_data_start)):
                    row_start = frame_data_start.iloc[landmark_index]
                    row_end = frame_data_end.iloc[landmark_index]
                    # Calculate interpolated values
                    interpolated_row = {
                        "row_id": row_start.row_id,
                        "landmark_index": row_start.landmark_index,
                        "x": row_start.x + t * (row_end.x - row_start.x),
                        "y": row_start.y + t * (row_end.y - row_start.y),
                        "z": row_start.z + t * (row_end.z - row_start.z),
                        "frame": new_frame_number,
                        "type": row_start.type,
                        "label": row_start.label,
                    }
                    interpolated_rows.append(interpolated_row)
                new_frame_number += 1  # Increment the frame number for the next set of interpolated frames

        df_interpolated = pd.DataFrame(interpolated_rows)

        df_combined = pd.concat([df, df_interpolated], ignore_index=True)
        df_combined.sort_values(by=["frame"], inplace=True)

        if df_combined["frame"].nunique() > self.target_frames:
            # Select evenly spaced frames
            selected_frames = np.round(
                np.linspace(
                    0, df_combined["frame"].max(), self.target_frames, endpoint=True
                )
            ).astype(int)
            df_combined = df_combined[df_combined["frame"].isin(selected_frames)]

        # Ensure landmark consistency
        df_combined = self._ensure_landmark_consistency(df_combined)

        return df_combined

    def _ensure_landmark_consistency(self, df):
        placeholder_landmark = {"x": -1, "y": -1, "z": 0}

        for frame_number in df["frame"].unique():
            frame_data = df[df["frame"] == frame_number]
            existing_landmarks = set(
                f"{row['type']}-{row['landmark_index']}"
                for _, row in frame_data.iterrows()
            )
            missing_landmarks = self.relevant_landmarks - existing_landmarks

            missing_landmark_data = [
                {
                    "frame": frame_number,
                    "type": lm.split("-")[0],
                    "landmark_index": int(lm.split("-")[1]),
                    **placeholder_landmark,
                    "label": "placeholder",
                }
                for lm in missing_landmarks
            ]

            if missing_landmark_data:
                df = pd.concat(
                    [df, pd.DataFrame(missing_landmark_data)], ignore_index=True
                )

        return df

    def _reduce_frames(self, df):
        """
        Reduces the number of frames in the dataframe to match the target number of frames.
        Ensures landmark consistency in each frame.

        :param df: The dataframe to reduce frames in.
        :return: The dataframe with reduced and consistent frames.
        """
        num_frames = len(df["frame"].unique())
        frames_to_keep = np.linspace(0, num_frames - 1, self.target_frames, dtype=int)
        df = df[df["frame"].isin(frames_to_keep)].copy()
        df["frame"] = df["frame"].rank(method="dense").astype(int) - 1

        # Ensure landmark consistency
        df = self._ensure_landmark_consistency(df)

        return df

    def _calculate_hand_features(self, df):
        """
        Calculates hand features for each frame in the dataframe.

        :param df: The dataframe to calculate hand features in.
        :return: The dataframe with hand features calculated.
        """
        # Initialize hand feature columns with NaN
        for hand in ["right_hand", "left_hand"]:
            df[f"{hand}_thumb_index_distance"] = np.nan
            df[f"{hand}_palm_orientation"] = np.nan

        for frame_number in df["frame"].unique():
            frame_data = df[df["frame"] == frame_number]
            for hand in ["right_hand", "left_hand"]:
                # Extract relevant landmarks
                wrist = frame_data[
                    (frame_data["type"] == hand) & (frame_data["landmark_index"] == 0)
                ][["x", "y"]].values
                thumb_tip = frame_data[
                    (frame_data["type"] == hand) & (frame_data["landmark_index"] == 4)
                ][["x", "y"]].values
                index_tip = frame_data[
                    (frame_data["type"] == hand) & (frame_data["landmark_index"] == 8)
                ][["x", "y"]].values
                pinky_tip = frame_data[
                    (frame_data["type"] == hand) & (frame_data["landmark_index"] == 20)
                ][["x", "y"]].values

                # Check if all required landmarks are present
                if (
                    wrist.size > 0
                    and thumb_tip.size > 0
                    and index_tip.size > 0
                    and pinky_tip.size > 0
                ):
                    wrist = wrist[0]
                    thumb_tip = thumb_tip[0]
                    index_tip = index_tip[0]
                    pinky_tip = pinky_tip[0]

                    # Calculate features
                    thumb_index_distance = self._calculate_distance(
                        thumb_tip, index_tip
                    )
                    palm_orientation = self._calculate_palm_orientation(
                        wrist, thumb_tip, pinky_tip
                    )

                    # Store results
                    df.loc[
                        (df["frame"] == frame_number) & (df["type"] == hand),
                        f"{hand}_thumb_index_distance",
                    ] = thumb_index_distance
                    df.loc[
                        (df["frame"] == frame_number) & (df["type"] == hand),
                        f"{hand}_palm_orientation",
                    ] = palm_orientation

        return df

    def _calculate_distance(self, point1, point2):
        """
        Calculates the Euclidean distance between two points.

        :param point1: The first point.
        :param point2: The second point.
        :return: The Euclidean distance between the two points.
        """
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def _calculate_angle(self, p1, p2, p3):
        """
        Calculates the angle between three points.

        :param p1: The first point.
        :param p2: The second point (vertex of the angle).
        :param p3: The third point.
        :return: The angle in degrees.
        """
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        ba = a - b
        bc = c - b

        # Check if norms of vectors are non-zero
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return np.nan  # Return NaN if one of the vectors is a zero vector

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def _calculate_palm_orientation(self, wrist, thumb_tip, pinky_tip):
        """
        Calculates the orientation of the palm based on wrist, thumb tip, and pinky tip positions.

        :param wrist: The position of the wrist.
        :param thumb_tip: The position of the thumb tip.
        :param pinky_tip: The position of the pinky tip.
        :return: The orientation of the palm in degrees.
        """
        mid_point = [
            (thumb_tip[0] + pinky_tip[0]) / 2,
            (thumb_tip[1] + pinky_tip[1]) / 2,
        ]
        angle = np.arctan2(mid_point[1] - wrist[1], mid_point[0] - wrist[0])
        return np.degrees(angle)

    def _calculate_finger_joint_angles(self, df):
        for hand in ["right_hand", "left_hand"]:
            for finger, joints in {
                "thumb": [1, 2, 3, 4],
                "index": [5, 6, 7, 8],
                "middle": [9, 10, 11, 12],
                "ring": [13, 14, 15, 16],
                "pinky": [17, 18, 19, 20],
            }.items():
                for i in range(3):
                    base = joints[i]
                    middle = joints[i + 1]
                    tip = joints[i + 2] if i < 2 else None

                    angle_name = f"{hand}_{finger}_{i}_angle"
                    df[angle_name] = np.nan

                    if tip is not None:
                        # Calculate angle between base, middle, and tip
                        df[angle_name] = df.apply(
                            lambda row: self._calculate_angle(
                                row.get(f"{hand}-{base}"),
                                row.get(f"{hand}-{middle}"),
                                row.get(f"{hand}-{tip}"),
                            )
                            if row.get(f"{hand}-{base}") is not None
                            and row.get(f"{hand}-{middle}") is not None
                            and row.get(f"{hand}-{tip}") is not None
                            else np.nan,
                            axis=1,
                        )
                    else:
                        # Calculate orientation angle
                        df[angle_name] = df.apply(
                            lambda row: self._calculate_orientation_angle(
                                row.get(f"{hand}-{base}"), row.get(f"{hand}-{middle}")
                            )
                            if row.get(f"{hand}-{base}") is not None
                            and row.get(f"{hand}-{middle}") is not None
                            else np.nan,
                            axis=1,
                        )
        return df

    def _calculate_finger_orientation_angles(self, df):
        for hand in ["right_hand", "left_hand"]:
            for finger, landmarks in {
                "thumb": [0, 4],
                "index": [0, 8],
                "middle": [0, 12],
                "ring": [0, 16],
                "pinky": [0, 20],
            }.items():
                base_index = landmarks[0]
                tip_index = landmarks[1]

                angle_name = f"{hand}_{finger}_orientation_angle"
                df[angle_name] = np.nan  # Initialize with NaN

                for frame in df["frame"].unique():
                    frame_data = df[(df["frame"] == frame) & (df["type"] == hand)]

                    base_landmark = frame_data[
                        frame_data["landmark_index"] == base_index
                    ][["x", "y", "z"]].values
                    tip_landmark = frame_data[
                        frame_data["landmark_index"] == tip_index
                    ][["x", "y", "z"]].values

                    if base_landmark.size > 0 and tip_landmark.size > 0:
                        base_landmark = base_landmark[0]
                        tip_landmark = tip_landmark[0]
                        angle = self._calculate_orientation_angle_3d(
                            base_landmark, tip_landmark
                        )
                        df.loc[
                            (df["frame"] == frame) & (df["type"] == hand), angle_name
                        ] = angle

        return df

    def _calculate_orientation_angle_3d(self, point_base, point_tip):
        # Define the reference vector (e.g., along the x-axis)
        reference_vector = np.array([1, 0, 0])

        # Create vectors from the points
        finger_vector = np.array(point_tip) - np.array(point_base)

        # Check if the norm of the finger vector is non-zero
        norm_finger_vector = np.linalg.norm(finger_vector)
        if norm_finger_vector == 0:
            return np.nan  # Return NaN if finger_vector is a zero vector

        # Normalize the vectors
        finger_vector_normalized = finger_vector / norm_finger_vector
        reference_vector_normalized = reference_vector / np.linalg.norm(
            reference_vector
        )

        # Calculate the angle using the dot product
        dot_product = np.dot(finger_vector_normalized, reference_vector_normalized)
        angle_rad = np.arccos(
            np.clip(dot_product, -1.0, 1.0)
        )  # Clip to handle numerical errors

        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def _calculate_orientation_angle(self, point1, point2):
        """
        Calculate the orientation angle of a line defined by two points relative to the horizontal line.

        :param point1: Coordinates of the first point (x1, y1)
        :param point2: Coordinates of the second point (x2, y2)
        :return: Orientation angle in degrees
        """
        dy = point2[1] - point1[1]
        dx = point2[0] - point1[0]
        angle = np.arctan2(dy, dx)
        return np.degrees(angle)

    def _calculate_arms_configuration(self, df):
        # Initialize the column as type 'object' to accommodate string values
        df["arms_configuration"] = None
        df["arms_configuration"] = df["arms_configuration"].astype("object")

        for frame_number in df["frame"].unique():
            frame_data = df[df["frame"] == frame_number]

            if self._are_arms_crossed(frame_data):
                config = "arms_crossed"
            elif self._are_arms_at_sides(frame_data):
                config = "arms_at_sides"
            elif self._is_one_arm_raised(frame_data):
                config = "one_arm_raised"
            elif self._are_both_arms_raised(frame_data):
                config = "both_arms_raised"
            elif self._are_arms_gesturing_to_body(frame_data):
                config = "arms_gesturing_to_body"
            else:
                config = "neutral"

            df.loc[df["frame"] == frame_number, "arms_configuration"] = config

        return df

    def _are_arms_crossed(self, frame_data):
        # Assuming 'frame_data' is a DataFrame with 'type', 'x', 'y', and landmark indices.

        # Check if the right and left wrists are present in the data
        right_wrist_data = frame_data.loc[
            (frame_data["type"] == "right_hand")
            & (frame_data.index == self.HAND_RIGHT_WRIST)
        ]
        left_wrist_data = frame_data.loc[
            (frame_data["type"] == "left_hand")
            & (frame_data.index == self.HAND_LEFT_WRIST)
        ]

        # If either wrist is not present, we cannot determine if arms are crossed
        if right_wrist_data.empty or left_wrist_data.empty:
            return False  # Or handle this case as needed

        # If both wrists are present, compare their x coordinates
        right_arm_x = right_wrist_data["x"].values[0]
        left_arm_x = left_wrist_data["x"].values[0]

        # Your logic to determine if arms are crossed
        # For example, assuming that if right_arm_x is less than left_arm_x, the arms are crossed
        are_crossed = right_arm_x < left_arm_x

        return are_crossed

    def _are_arms_at_sides(self, frame_data):
        # Extract the x coordinates for shoulders and wrists if they exist
        right_shoulder_x = frame_data[
            (frame_data["type"] == "pose")
            & (frame_data["landmark_index"] == self.POSE_RIGHT_SHOULDER)
        ]["x"].values
        left_shoulder_x = frame_data[
            (frame_data["type"] == "pose")
            & (frame_data["landmark_index"] == self.POSE_LEFT_SHOULDER)
        ]["x"].values
        right_wrist_x = frame_data[
            (frame_data["type"] == "right_hand")
            & (frame_data["landmark_index"] == self.HAND_RIGHT_WRIST)
        ]["x"].values
        left_wrist_x = frame_data[
            (frame_data["type"] == "left_hand")
            & (frame_data["landmark_index"] == self.HAND_LEFT_WRIST)
        ]["x"].values

        # Check if the required landmarks are present
        if (
            right_shoulder_x.size > 0
            and left_shoulder_x.size > 0
            and right_wrist_x.size > 0
            and left_wrist_x.size > 0
        ):
            shoulders_width = abs(right_shoulder_x[0] - left_shoulder_x[0])
            right_wrist_distance = abs(right_wrist_x[0] - right_shoulder_x[0])
            left_wrist_distance = abs(left_wrist_x[0] - left_shoulder_x[0])

            # Assuming that if wrists are roughly below shoulders, arms are at sides
            return (
                right_wrist_distance < shoulders_width
                and left_wrist_distance < shoulders_width
            )
        else:
            return False

    def _is_one_arm_raised(self, frame_data):
        # Filter for right and left elbow 'y' values
        right_elbow_data = frame_data[
            (frame_data["type"] == "pose")
            & (frame_data["landmark_index"] == self.POSE_RIGHT_ELBOW)
        ]
        left_elbow_data = frame_data[
            (frame_data["type"] == "pose")
            & (frame_data["landmark_index"] == self.POSE_LEFT_ELBOW)
        ]

        # Filter for right and left shoulder 'y' values
        right_shoulder_data = frame_data[
            (frame_data["type"] == "pose")
            & (frame_data["landmark_index"] == self.POSE_RIGHT_SHOULDER)
        ]
        left_shoulder_data = frame_data[
            (frame_data["type"] == "pose")
            & (frame_data["landmark_index"] == self.POSE_LEFT_SHOULDER)
        ]

        # Initialize flags
        right_arm_raised = False
        left_arm_raised = False

        # Check if the right elbow is above the right shoulder
        if not right_elbow_data.empty and not right_shoulder_data.empty:
            right_elbow_y = right_elbow_data["y"].values[0]
            right_shoulder_y = right_shoulder_data["y"].values[0]
            right_arm_raised = right_elbow_y < right_shoulder_y

        # Check if the left elbow is above the left shoulder
        if not left_elbow_data.empty and not left_shoulder_data.empty:
            left_elbow_y = left_elbow_data["y"].values[0]
            left_shoulder_y = left_shoulder_data["y"].values[0]
            left_arm_raised = left_elbow_y < left_shoulder_y

        # Return true if either arm is raised
        return right_arm_raised or left_arm_raised

    def _are_both_arms_raised(self, frame_data):
        # Check if the required landmarks are present
        try:
            right_elbow_y = frame_data.loc[
                (frame_data["type"] == "pose")
                & (frame_data["landmark_index"] == self.POSE_RIGHT_ELBOW),
                "y",
            ].iloc[0]
            left_elbow_y = frame_data.loc[
                (frame_data["type"] == "pose")
                & (frame_data["landmark_index"] == self.POSE_LEFT_ELBOW),
                "y",
            ].iloc[0]
            right_shoulder_y = frame_data.loc[
                (frame_data["type"] == "pose")
                & (frame_data["landmark_index"] == self.POSE_RIGHT_SHOULDER),
                "y",
            ].iloc[0]
            left_shoulder_y = frame_data.loc[
                (frame_data["type"] == "pose")
                & (frame_data["landmark_index"] == self.POSE_LEFT_SHOULDER),
                "y",
            ].iloc[0]
        except (
            IndexError
        ):  # If any of the landmarks is missing, we catch the IndexError
            return False  # Can't determine if both arms are raised, so we return False

        # Assuming that if both elbows are above shoulder level, both arms are raised
        return right_elbow_y < right_shoulder_y and left_elbow_y < left_shoulder_y

    def _are_arms_gesturing_to_body(self, frame_data):
        try:
            right_wrist_x = frame_data.loc[
                (frame_data["type"] == "right_hand")
                & (frame_data["landmark_index"] == self.HAND_RIGHT_WRIST),
                "x",
            ].iloc[0]
            left_wrist_x = frame_data.loc[
                (frame_data["type"] == "left_hand")
                & (frame_data["landmark_index"] == self.HAND_LEFT_WRIST),
                "x",
            ].iloc[0]
            nose_x = frame_data.loc[
                (frame_data["type"] == "pose")
                & (frame_data["landmark_index"] == self.POSE_NOSE),
                "x",
            ].iloc[0]
        except (
            IndexError
        ):  # If any of the landmarks is missing, we catch the IndexError
            return False  # Can't determine if arms are gesturing to body, so we return False

        # Assuming that if wrists are horizontally inside the body silhouette towards the nose, arms are gesturing
        # to the body
        return right_wrist_x > nose_x > left_wrist_x

    def _calculate_wrist_features(self, df):
        """
        Calculates wrist-to-wrist distance and angle for each frame in the dataframe.

        :param df: The dataframe to calculate wrist features in.
        :return: The dataframe with wrist features calculated.
        """
        df["wrist_to_wrist_distance"] = np.nan
        df["wrist_to_wrist_angle"] = np.nan

        for frame_number in df["frame"].unique():
            frame_data = df[df["frame"] == frame_number]

            right_wrist = frame_data[
                (frame_data["type"] == "right_hand")
                & (frame_data["landmark_index"] == 0)
            ][["x", "y"]].values
            left_wrist = frame_data[
                (frame_data["type"] == "left_hand")
                & (frame_data["landmark_index"] == 0)
            ][["x", "y"]].values

            if right_wrist.size > 0 and left_wrist.size > 0:
                right_wrist = right_wrist[0]
                left_wrist = left_wrist[0]

                distance = self._calculate_distance(right_wrist, left_wrist)
                angle = self._calculate_orientation_angle(right_wrist, left_wrist)

                df.loc[
                    df["frame"] == frame_number, "wrist_to_wrist_distance"
                ] = distance
                df.loc[df["frame"] == frame_number, "wrist_to_wrist_angle"] = angle

        return df

    # Temporal Features

    def _calculate_temporal_features(self, df):
        # First, ensure the dataframe is sorted by frame
        df = df.sort_values(by=["frame", "landmark_index"]).reset_index(drop=True)

        # Initialize columns for velocity and acceleration
        for coord in ["x", "y"]:
            df[f"velocity_{coord}"] = np.nan
            df[f"acceleration_{coord}"] = np.nan

        # Initialize velocity and acceleration at frame 0 to zero for all landmarks
        for coord in ["x", "y"]:
            df.loc[df["frame"] == 0, f"velocity_{coord}"] = 0
            df.loc[df["frame"] == 0, f"acceleration_{coord}"] = 0

        # Get the list of unique frames to loop through
        unique_frames = df["frame"].unique()

        # Loop through each pair of consecutive frames
        for i in range(len(unique_frames) - 1):
            current_frame = unique_frames[i]
            next_frame = unique_frames[i + 1]

            # Get the landmarks for the current and next frame
            current_frame_landmarks = df[df["frame"] == current_frame]
            next_frame_landmarks = df[df["frame"] == next_frame]

            # Find common landmarks by index and type
            common_landmarks = set(
                zip(
                    current_frame_landmarks["landmark_index"],
                    current_frame_landmarks["type"],
                )
            ).intersection(
                set(
                    zip(
                        next_frame_landmarks["landmark_index"],
                        next_frame_landmarks["type"],
                    )
                )
            )

            # Calculate temporal features for common landmarks
            for landmark_index, landmark_type in common_landmarks:
                current_landmark_data = current_frame_landmarks[
                    (current_frame_landmarks["landmark_index"] == landmark_index)
                    & (current_frame_landmarks["type"] == landmark_type)
                ]
                next_landmark_data = next_frame_landmarks[
                    (next_frame_landmarks["landmark_index"] == landmark_index)
                    & (next_frame_landmarks["type"] == landmark_type)
                ]

                # Ensure there is only one row per landmark per frame
                if len(current_landmark_data) == 1 and len(next_landmark_data) == 1:
                    for coord in ["x", "y"]:
                        # Calculate velocity
                        velocity = (
                            next_landmark_data[coord].values[0]
                            - current_landmark_data[coord].values[0]
                        )
                        df.loc[next_landmark_data.index, f"velocity_{coord}"] = velocity

                        # Calculate acceleration
                        if i > 0:
                            prev_velocity = df.loc[
                                current_landmark_data.index, f"velocity_{coord}"
                            ].values[0]
                            acceleration = velocity - prev_velocity
                            df.loc[
                                next_landmark_data.index, f"acceleration_{coord}"
                            ] = acceleration
                        else:
                            # For the second frame (i == 0), set acceleration to zero
                            df.loc[
                                next_landmark_data.index, f"acceleration_{coord}"
                            ] = 0

        return df

    def _calculate_all_hand_configurations(self, df):
        """
        Calculate all hand configurations for each frame and add them to the dataframe.

        :param df: The dataframe to process.
        :return: The dataframe with new columns for hand configurations.
        """
        # Define the hand configurations as columns with default values
        hand_config_cols = [
            "fist_score",
            "flat_hand_score",
            "open_hand_score",
            "one_finger_extended_score",
            "two_fingers_extended_score",
            "hook_score",
            "cup_score",
            "pinch_score",
            "thumb_exposed_score",
        ]

        # Initialize the columns for hand configurations
        for col in hand_config_cols:
            df[col] = np.nan  # Using NaN for uninitialized scores

        # Iterate over the frames and calculate the hand configurations
        for frame_number in df["frame"].unique():
            frame_data = df[df["frame"] == frame_number]

            # For each hand, calculate the scores for all configurations
            for hand in ["right_hand", "left_hand"]:
                # Ensure we have data for the hand
                if hand in frame_data["type"].values:
                    hand_data = frame_data[frame_data["type"] == hand]
                    df.loc[
                        df["frame"] == frame_number, f"{hand}_fist_score"
                    ] = self._calculate_fist_score(hand_data)
                    df.loc[
                        df["frame"] == frame_number, f"{hand}_flat_hand_score"
                    ] = self._calculate_flat_hand_score(hand_data)
                    df.loc[
                        df["frame"] == frame_number, f"{hand}_open_hand_score"
                    ] = self._calculate_open_hand_score(hand_data)
                    df.loc[
                        df["frame"] == frame_number, f"{hand}_one_finger_extended_score"
                    ] = self._calculate_one_finger_extended_score(hand_data)
                    df.loc[
                        df["frame"] == frame_number,
                        f"{hand}_two_fingers_extended_score",
                    ] = self._calculate_two_fingers_extended_score(hand_data)
                    df.loc[
                        df["frame"] == frame_number, f"{hand}_hook_score"
                    ] = self._calculate_hook_score(hand_data)
                    df.loc[
                        df["frame"] == frame_number, f"{hand}_cup_score"
                    ] = self._calculate_cup_score(hand_data)
                    df.loc[
                        df["frame"] == frame_number, f"{hand}_pinch_score"
                    ] = self._calculate_pinch_score(hand_data)
                    df.loc[
                        df["frame"] == frame_number, f"{hand}_thumb_exposed_score"
                    ] = self._calculate_thumb_exposed_score(hand_data)

        return df

    def _calculate_fist_score(self, df):
        # Extract palm base and all fingertips data
        palm_base_data = df[df["landmark_index"] == 0][["x", "y", "z"]]
        fingertips_data = df[df["landmark_index"].isin([4, 8, 12, 16, 20])][
            ["x", "y", "z"]
        ]
        epsilon = 1e-6

        if not palm_base_data.empty and not fingertips_data.empty:
            palm_base = palm_base_data.values[0]
            fingertips = fingertips_data.values
            distances = np.sqrt(np.sum((fingertips - palm_base) ** 2, axis=1))
            # Invert the distances to have higher scores for closer fingertips
            inverted_distances = 1 / (distances + epsilon)
            fist_score = np.mean(inverted_distances)
        else:
            return np.nan  # Return NaN if there's missing data

        # Normalization with new limits based on observed data for a full fist
        min_limit = np.min(
            inverted_distances
        )  # Min inverted distance (farthest fingertip)
        max_limit = np.max(
            inverted_distances
        )  # Max inverted distance (closest fingertip)
        # Add an epsilon to the denominator to avoid division by zero

        normalized_fist_score = (fist_score - min_limit) / (
            max_limit - min_limit + epsilon
        )

        # Scale the normalized score from 0 to 1 to be from -1 to 1
        normalized_fist_score = 2 * normalized_fist_score - 1

        return normalized_fist_score

    def _calculate_flat_hand_score(self, df):
        fingertips_z = df[df["landmark_index"].isin([4, 8, 12, 16, 20])]["z"].values
        if len(fingertips_z) == 5:
            variance_z = np.var(fingertips_z)
            # Normalize the variance to have a higher score for lower variance
            # Assuming the observed maximum variance for a non-flat hand is a pre-determined value
            max_variance = 0.05  # This is an example value and should be adjusted based on your data
            flat_hand_score = max(0, 1 - variance_z / max_variance)
            # Scale from 0 to 1 to -1 to 1
            normalized_flat_hand_score = 2 * flat_hand_score - 1
            return normalized_flat_hand_score
        else:
            return np.nan  # Return NaN if not all fingertips are present

    def _calculate_open_hand_score(self, df):
        fingertips = df[df["landmark_index"].isin([4, 8, 12, 16, 20])][
            ["x", "y"]
        ].values
        if len(fingertips) == 5:
            distances = []
            for i in range(len(fingertips)):
                for j in range(i + 1, len(fingertips)):
                    distances.append(np.linalg.norm(fingertips[i] - fingertips[j]))
            average_distance = np.mean(distances)
            # Normalize the average distance to have a higher score for greater distances
            # Assuming the observed maximum average distance for a fully open hand is a pre-determined value
            max_distance = 0.3  # This is an example value and should be adjusted based on your data
            open_hand_score = min(1, average_distance / max_distance)
            # Scale from 0 to 1 to -1 to 1
            normalized_open_hand_score = 2 * open_hand_score - 1
            return normalized_open_hand_score
        else:
            return np.nan  # Return NaN if not all fingertips are present

    def _calculate_finger_extension_score(self, finger_tip, palm_base):
        """
        Calculate the extension score for a single finger.
        Score ranges from -1 (not extended) to +1 (fully extended).
        """
        fully_bent_distance = 0.05  # Placeholder for a fully bent finger
        fully_extended_distance = 0.25  # Placeholder for a fully extended finger

        distance = np.linalg.norm(finger_tip - palm_base)

        # Normalize the distance to a score of -1 to 1
        if distance < fully_bent_distance:
            return -1
        elif distance > fully_extended_distance:
            return 1
        else:
            return (
                2
                * (
                    (distance - fully_bent_distance)
                    / (fully_extended_distance - fully_bent_distance)
                )
                - 1
            )

    def _calculate_one_finger_extended_score(self, df):
        fingertips_data = df[df["landmark_index"].isin([8, 12, 16, 20])][
            ["x", "y", "z"]
        ].values
        palm_base_data = df[df["landmark_index"] == 0][["x", "y", "z"]].values
        if fingertips_data.shape[0] == 4 and palm_base_data.size == 3:
            palm_base = palm_base_data[0]
            extension_scores = [
                self._calculate_finger_extension_score(finger_tip, palm_base)
                for finger_tip in fingertips_data
            ]
            extended_fingers_count = sum(
                score > 0 for score in extension_scores
            )  # Adjusted threshold
            return (
                1 if extended_fingers_count == 1 else -1
            )  # 1 for exactly one extended finger
        else:
            return np.nan  # Incomplete data

    def _calculate_two_fingers_extended_score(self, df):
        fingertips_data = df[df["landmark_index"].isin([8, 12, 16, 20])][
            ["x", "y", "z"]
        ].values
        palm_base_data = df[df["landmark_index"] == 0][["x", "y", "z"]].values
        if fingertips_data.shape[0] == 4 and palm_base_data.size == 3:
            palm_base = palm_base_data[0]
            extension_scores = [
                self._calculate_finger_extension_score(finger_tip, palm_base)
                for finger_tip in fingertips_data
            ]
            extended_fingers_count = sum(
                score > 0 for score in extension_scores
            )  # Adjusted threshold
            return (
                1 if extended_fingers_count == 2 else -1
            )  # 1 for exactly two extended fingers
        else:
            return np.nan  # Incomplete data

    def _calculate_hook_score(self, df):
        angles = []
        finger_distances = []
        thumb_influence = self._thumb_influence(df)

        for finger_base in [5, 9, 13, 17]:  # Excluding the thumb
            fingertip_data = df[df["landmark_index"] == finger_base + 3][
                ["x", "y", "z"]
            ]
            middle_joint_data = df[df["landmark_index"] == finger_base + 2][
                ["x", "y", "z"]
            ]
            base_joint_data = df[df["landmark_index"] == finger_base + 1][
                ["x", "y", "z"]
            ]
            if (
                fingertip_data.size == 3
                and middle_joint_data.size == 3
                and base_joint_data.size == 3
            ):
                fingertip = fingertip_data.values[0]
                middle_joint = middle_joint_data.values[0]
                base_joint = base_joint_data.values[0]
                angle = self._calculate_angle(fingertip, middle_joint, base_joint)
                angles.append(angle)
                finger_distances.extend(self._calculate_finger_distances(df, fingertip))

        # Adjust the sensitivity of the hook score
        mean_angle = np.mean(angles)
        adjusted_hook_score = self._scale_angle_to_score(
            mean_angle, min_angle=10, max_angle=90
        )

        # Incorporate finger spacing into the score
        avg_finger_distance = np.mean(finger_distances)
        spacing_factor = max(
            0, 1 - avg_finger_distance / 0.15
        )  # Adjust 0.15 based on data

        # Combine thumb influence, adjusted hook score, and spacing factor
        combined_score = (
            (0.4 * thumb_influence)
            + (0.4 * adjusted_hook_score)
            + (0.2 * spacing_factor)
        )

        # Normalize and scale the score to be between -1 and 1
        normalized_hook_score = max(-1, min(1, 2 * combined_score - 1))
        return normalized_hook_score

    def _thumb_influence(self, df):
        thumb_tip_data = df[df["landmark_index"] == 4][["x", "y", "z"]]
        palm_base_data = df[df["landmark_index"] == 0][["x", "y", "z"]]

        if not thumb_tip_data.empty and not palm_base_data.empty:
            thumb_tip = thumb_tip_data.values[0]
            palm_base = palm_base_data.values[0]
            distance = np.linalg.norm(thumb_tip - palm_base)

            # Adjust these thresholds based on data
            max_distance_for_influence = 0.3
            min_distance_for_influence = 0.1

            if distance < min_distance_for_influence:
                return 1  # Maximum thumb influence
            elif distance > max_distance_for_influence:
                return 0  # No thumb influence
            else:
                # Scale the influence linearly based on the distance
                return 1 - (
                    (distance - min_distance_for_influence)
                    / (max_distance_for_influence - min_distance_for_influence)
                )
        else:
            return 0

    def _calculate_finger_distances(self, df, reference_fingertip):
        distances = []
        for fingertip_index in [8, 12, 16, 20]:  # Indices of the other fingertips
            fingertip_data = df[df["landmark_index"] == fingertip_index][
                ["x", "y", "z"]
            ]
            if not fingertip_data.empty:
                fingertip = fingertip_data.values[0]
                distance = np.linalg.norm(fingertip - reference_fingertip)
                distances.append(distance)
        return distances

    def _evaluate_finger_curl_for_hook(self, df, palm_base, max_distance):
        """
        Evaluate the curvature of fingers towards the palm for hooking.
        More sensitive to partial curls typical in a hook.
        """
        curl_score = 0
        for fingertip_index in [8, 12, 16, 20]:  # Fingertips indices
            fingertip_data = df[df["landmark_index"] == fingertip_index][
                ["x", "y", "z"]
            ]
            if not fingertip_data.empty:
                fingertip = fingertip_data.values[0]
                distance = np.linalg.norm(fingertip - palm_base)
                # Adjust the scoring for partial curls
                partial_curl_score = max(0, (1 - distance / max_distance) ** 2)
                curl_score += partial_curl_score
        return curl_score / 4  # Normalize by the number of fingers

    def _scale_angle_to_score(self, angle, min_angle=60, max_angle=90):
        # min_angle = 60  # Adjusted min angle for a non-hooked finger
        # max_angle = 120  # Adjusted max angle for a fully hooked finger

        if angle < min_angle:
            return -1
        elif angle > max_angle:
            return 1
        else:
            return 2 * ((angle - min_angle) / (max_angle - min_angle)) - 1

    def _calculate_cup_score(self, df):
        palm_base_data = df[df["landmark_index"] == 0][["x", "y", "z"]]
        thumb_tip_data = df[df["landmark_index"] == 4][["x", "y", "z"]]
        if palm_base_data.size == 3 and thumb_tip_data.size == 3:
            palm_base = palm_base_data.values[0]
            thumb_tip = thumb_tip_data.values[0]

            # Significantly increase the maximum allowable distances
            thumb_proximity_score = self._evaluate_thumb_proximity(
                thumb_tip, palm_base, max_distance=0.5
            )
            finger_curl_score = self._evaluate_finger_curl(
                df, palm_base, max_distance=0.5
            )

            # Heavily weight the finger closeness and curl scores
            finger_closeness_score = self._evaluate_finger_closeness(
                df, max_avg_distance=0.4
            )

            # Adjust the weights for component scores
            combined_score = (
                0.1 * thumb_proximity_score
                + 0.45 * finger_closeness_score
                + 0.45 * finger_curl_score
            )

            # Normalize and scale the score to be between -1 and 1
            normalized_cup_score = max(-1, min(1, 2 * combined_score - 1))
            return normalized_cup_score
        else:
            return np.nan

    def _evaluate_thumb_proximity(self, thumb_tip, palm_base, max_distance):
        """
        Evaluate the proximity of the thumb to the palm base for cupping.
        """
        distance = np.linalg.norm(thumb_tip - palm_base)
        return max(0, 1 - distance / max_distance)

    def _evaluate_finger_curl(self, df, palm_base, max_distance):
        """
        Evaluate the curvature of fingers towards the palm for cupping.
        """
        curl_score = 0
        for fingertip_index in [8, 12, 16, 20]:
            fingertip_data = df[df["landmark_index"] == fingertip_index][
                ["x", "y", "z"]
            ]
            if not fingertip_data.empty:
                fingertip = fingertip_data.values[0]
                distance = np.linalg.norm(fingertip - palm_base)
                curl_score += max(0, 1 - distance / max_distance)
        return curl_score / 4  # Normalize by the number of fingers

    def _evaluate_finger_closeness(self, df, max_avg_distance):
        """
        Evaluate the closeness of fingers for cupping.
        """
        total_distance = 0
        count = 0
        for i in range(8, 20, 4):  # Fingertip indices excluding the thumb
            fingertip_data = df[df["landmark_index"] == i][["x", "y", "z"]]
            if fingertip_data.size == 3:
                for j in range(i + 4, 21, 4):
                    next_fingertip_data = df[df["landmark_index"] == j][["x", "y", "z"]]
                    if next_fingertip_data.size == 3:
                        next_fingertip = next_fingertip_data.values[0]
                        fingertip = fingertip_data.values[0]
                        distance = np.linalg.norm(fingertip - next_fingertip)
                        total_distance += distance
                        count += 1
        if count > 0:
            avg_distance = total_distance / count
            return max(0, 1 - avg_distance / max_avg_distance)
        else:
            return 0

    def _calculate_pinch_score(self, df):
        thumb_tip_data = df[df["landmark_index"] == 4][["x", "y", "z"]]
        if thumb_tip_data.empty:
            return np.nan

        thumb_tip = thumb_tip_data.values[0]
        pinch_scores = []

        # Adjust proximity threshold based on your dataset
        proximity_threshold = 0.05  # Example threshold, needs to be adjusted

        for fingertip_index in [8, 12]:  # Focusing on index and middle fingers
            fingertip_data = df[df["landmark_index"] == fingertip_index][
                ["x", "y", "z"]
            ]
            if not fingertip_data.empty:
                fingertip = fingertip_data.values[0]
                distance_to_thumb = np.linalg.norm(fingertip - thumb_tip)

                # Adjust curl evaluation
                curl_score = self._calculate_curl_score_for_pinch(fingertip, thumb_tip)

                # Scoring based on proximity and adjusted curl
                if distance_to_thumb < proximity_threshold:
                    score = (1 - distance_to_thumb / proximity_threshold) + curl_score
                else:
                    score = 0  # No pinch if the distance is greater than the threshold

                pinch_scores.append(score)

        # Simple average of pinch scores for index and middle fingers
        if pinch_scores:
            return np.mean(pinch_scores)
        else:
            return np.nan

    def _calculate_curl_score_for_pinch(self, fingertip, thumb_tip):
        """
        Calculate a specialized curl score for a finger in the context of a pinch.
        The score should reflect how much the finger is curled towards the thumb.
        """
        # Example implementation: Use the distance between the fingertip and thumb tip.
        # Smaller distance indicates a stronger pinch, resulting in a higher score.
        distance = np.linalg.norm(fingertip - thumb_tip)

        # Define a suitable threshold for your dataset.
        max_pinch_distance = 0.05  # This is an example value, needs to be adjusted

        # Calculate the score
        if distance < max_pinch_distance:
            # The closer the finger is to the thumb, the higher the score
            curl_score = 1 - (distance / max_pinch_distance)
        else:
            # No pinch if the distance is greater than the threshold
            curl_score = 0

        return curl_score

    def _calculate_thumb_exposed_score(self, df):
        thumb_tip_data = df[df["landmark_index"] == 4][["x", "y", "z"]]
        thumb_cmc_data = df[df["landmark_index"] == 1][["x", "y", "z"]]
        wrist_data = df[df["landmark_index"] == 0][["x", "y", "z"]]

        if (
            not thumb_tip_data.empty
            and not thumb_cmc_data.empty
            and not wrist_data.empty
        ):
            thumb_tip = thumb_tip_data.values[0]
            thumb_cmc = thumb_cmc_data.values[0]
            wrist = wrist_data.values[0]

            # Distance between thumb tip and CMC
            thumb_dist = np.linalg.norm(thumb_tip - thumb_cmc)
            # Calculate angle between thumb tip, CMC, and wrist for thumb exposure
            thumb_angle = self._calculate_angle(thumb_tip, thumb_cmc, wrist)
            # Combine distance and angle to calculate the thumb exposure score
            thumb_exposure_score = thumb_dist * thumb_angle

            # Normalize the score using a suitable range for the combined score
            min_limit = 0  # Assuming minimum exposure score
            max_limit = np.max(
                [thumb_dist * 180, 1]
            )  # Max possible score for fully extended thumb
            normalized_thumb_exposure_score = (thumb_exposure_score - min_limit) / (
                max_limit - min_limit
            )
            # Scale the normalized score from 0 to 1 to be from -1 to 1
            normalized_thumb_exposure_score = 2 * normalized_thumb_exposure_score - 1
            return normalized_thumb_exposure_score
        else:
            return np.nan  # Return NaN if any landmark data is missing

    def _format_example(self, df, parquet_file):
        """
        Formats a single example into the required output format.

        :param df: The dataframe representing a single example.
        :param parquet_file: The parquet file containing the original data.
        :return: A dictionary representing the formatted example.
        """

        hand_config_cols = [
            "fist_score",
            "flat_hand_score",
            "open_hand_score",
            "one_finger_extended_score",
            "two_fingers_extended_score",
            "hook_score",
            "cup_score",
            "pinch_score",
            "thumb_exposed_score",
        ]

        frames = []
        sequential_frame_number = (
            0  # Initialize a variable to track the frame number sequentially
        )

        for _, frame_data in df.groupby("frame"):
            frame_info = {
                "frame": sequential_frame_number,  # Use the sequential frame number instead of the original
                "landmarks": [],
                "spatial": {
                    "arms_configuration": frame_data["arms_configuration"].iloc[0]
                },
                "temporal": [],
            }

            for landmark_type, landmark_data in frame_data.groupby("type"):
                # Iterating through each row in the landmark data
                for idx, row in landmark_data.iterrows():
                    # Creating a dictionary for each landmark with 'landmark', 'x', and 'y' keys
                    landmark_dict = {
                        "landmark": f"{landmark_type}-{int(row['landmark_index'])}",
                        "x": row["x"],
                        "y": row["y"]
                    }
                    # Adding this dictionary to the 'landmarks' list in 'frame_info'
                    frame_info["landmarks"].append(landmark_dict)
                # landmarks = landmark_data[["x", "y"]].values.tolist()
                # frame_info["landmarks"].extend(landmarks)
                # frame_info["landmark_types"].extend(
                #     [
                #         f"{landmark_type}-{int(idx)}"
                #         for idx in landmark_data["landmark_index"]
                #     ]
                # )

                # Check if velocity and acceleration columns are present
                if "velocity_x" in landmark_data and "velocity_y" in landmark_data:
                    for _, row in landmark_data.iterrows():
                        # Skip the temporal data if velocity or acceleration is NaN
                        if (
                            pd.isna(row["velocity_x"])
                            or pd.isna(row["velocity_y"])
                            or pd.isna(row["acceleration_x"])
                            or pd.isna(row["acceleration_y"])
                        ):
                            continue

                        frame_info["temporal"].append(
                            {
                                "landmark": f"{landmark_type}-{int(row['landmark_index'])}",
                                "velocity": {
                                    "x": row["velocity_x"],
                                    "y": row["velocity_y"],
                                },
                                "acceleration": {
                                    "x": row["acceleration_x"],
                                    "y": row["acceleration_y"],
                                },
                            }
                        )

            # Add hand configurations to spatial features for this frame
            for col in ["right_hand_configuration", "left_hand_configuration"]:
                if col in frame_data.columns and not frame_data[col].isnull().all():
                    frame_info["spatial"][col] = frame_data[col].iloc[0]

            # Add hand features
            for hand in ["right_hand", "left_hand"]:
                thumb_index_distance = frame_data.loc[
                    frame_data["type"] == hand, f"{hand}_thumb_index_distance"
                ].values
                palm_orientation = frame_data.loc[
                    frame_data["type"] == hand, f"{hand}_palm_orientation"
                ].values

                if thumb_index_distance.size > 0 and not pd.isnull(
                    thumb_index_distance[0]
                ):
                    frame_info["spatial"][
                        f"{hand}_thumb_index_distance"
                    ] = thumb_index_distance[0]
                if palm_orientation.size > 0 and not pd.isnull(palm_orientation[0]):
                    frame_info["spatial"][
                        f"{hand}_palm_orientation"
                    ] = palm_orientation[0]

                # Add finger orientation angles
                for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                    angle_name = f"{hand}_{finger}_orientation_angle"
                    angle_value = frame_data.loc[
                        frame_data["type"] == hand, angle_name
                    ].values
                    if angle_value.size > 0 and not pd.isnull(angle_value[0]):
                        frame_info["spatial"][angle_name] = angle_value[0]

            # Add hand configuration scores to spatial features for this frame
            for hand in ["right_hand", "left_hand"]:
                for config in hand_config_cols:
                    hand_config_column = f"{hand}_{config}"
                    if hand_config_column in df.columns:
                        score = frame_data[hand_config_column].iloc[0]
                        if not pd.isnull(score):
                            frame_info["spatial"][hand_config_column] = score

            # Remove the "spatial" key if it's empty
            if not frame_info["spatial"]:
                frame_info.pop("spatial")

            frames.append(frame_info)
            sequential_frame_number += (
                1  # Increment the sequential frame number for the next frame
            )

        result = {"frames": frames, "file": parquet_file}
        return result

    def process(self):
        """
        Processes all signs and saves the cleaned and formatted data to disk.
        """
        all_signs_data = {}

        # Set Pandas options to prevent truncation
        pd.set_option("display.max_rows", None)  # Show all rows
        pd.set_option("display.max_columns", None)  # Show all columns
        pd.set_option("display.width", None)  # Use maximum width for displaying
        pd.set_option("display.max_colwidth", None)  # Show full content of each column

        for sign in tqdm(self.signs_to_process, desc="Processing signs", unit="sign"):
            all_signs_data[sign] = {"sign": sign, "examples": []}
            parquet_files = self._filter_files_by_sign(sign)
            print(f"Found {len(parquet_files)} parquet files for sign {sign}")

            for parquet_file in tqdm(
                parquet_files, desc=f"Cleaning [{sign}]", unit="file"
            ):
                df = pd.read_parquet(parquet_file)
                df = self._remove_empty_frames(df)
                df["sign"] = sign
                df = self._extract_relevant_landmarks(df)
                df = df.sort_values(by=["frame", "landmark_index"]).reset_index(
                    drop=True
                )
                df = self._handle_nan_values(df)

                if sign in self.label_map:
                    df["label"] = self.label_map[sign]
                else:
                    print(f"Warning: '{sign}' not found in label map. Skipping...")
                    continue

                df = self._interpolate_frames(df)
                df = self._calculate_all_hand_configurations(df)

                # Calculate temporal features
                df = self._calculate_temporal_features(df)

                df = self._calculate_hand_features(df)
                df = self._calculate_wrist_features(df)
                df = self._calculate_finger_joint_angles(df)
                df = self._calculate_finger_orientation_angles(df)
                df = self._calculate_arms_configuration(df)

                df = self._drop_z_coordinate(df)
                df = self._normalize_coordinates(df)
                df = self._smooth_landmarks(df, window_length=5, polyorder=3)

                example = self._format_example(df, parquet_file)

                if example is not None:
                    all_signs_data[sign]["examples"].append(example)
                else:
                    print(
                        f"Warning: No data returned for sign '{sign}' from file '{parquet_file}'."
                    )

            output_filename = os.path.join(
                self.base_dir, f"processed-{self.target_frames}-{self.max_files_per_sign}/{sign}.json"
            )

            output_dir = os.path.join(self.base_dir, f"processed-{self.target_frames}-{self.max_files_per_sign}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_filename, "w") as f:
                json.dump(all_signs_data[sign], f, indent=2)

            print(
                f"Data for sign '{sign}' has been cleaned and saved to {output_filename}"
            )


def main():
    """
    Main function to run the SLGraphDataBuilder.
    """
    load_dotenv()
    BASE_DIR = os.getenv("ASL_SIGNS_BASE_DIRECTORY")
    SIGNS_TO_PROCESS = [
        "TV",
        "after",
        "airplane",
        "all",
        "alligator",
        "animal",
        "another",
        "any",
        "apple",
        "arm",
        "aunt",
        "awake",
        "backyard",
        "bad",
        "balloon",
        "bath",
        "because",
        "bed",
        "bedroom",
        "bee",
        "before",
        "beside",
        "better",
        "bird",
        "black",
        "blow",
        "blue",
        "boat",
        "book",
        "boy",
        "brother",
        "brown",
        "bug",
        "bye",
        "callonphone",
        "can",
        "car",
        "carrot",
        "cat",
        "cereal",
        "chair",
        "cheek",
        "child",
        "chin",
        "chocolate",
        "clean",
        "close",
        "closet",
        "cloud",
        "clown",
        "cow",
        "cowboy",
        "cry",
        "cut",
        "cute",
        "dad",
        "dance",
        "dirty",
        "dog",
        "doll",
        "donkey",
        "down",
        "drawer",
        "drink",
        "drop",
        "dry",
        "dryer",
        "duck",
        "ear",
        "elephant",
        "empty",
        "every",
        "eye",
        "face",
        "fall",
        "farm",
        "fast",
        "feet",
        "find",
        "fine",
        "finger",
        "finish",
        "fireman",
        "first",
        "fish",
        "flag",
        "flower",
        "food",
        "for",
        "frenchfries",
        "frog",
        "garbage",
        "gift",
        "giraffe",
        "girl",
        "give",
        "glasswindow",
        "go",
        "goose",
        "grandma",
        "grandpa",
        "grass",
        "green",
        "gum",
        "hair",
        "happy",
        "hat",
        "hate",
        "have",
        "haveto",
        "head",
        "hear",
        "helicopter",
        "hello",
        "hen",
        "hesheit",
        "hide",
        "high",
        "home",
        "horse",
        "hot",
        "hungry",
        "icecream",
        "if",
        "into",
        "jacket",
        "jeans",
        "jump",
        "kiss",
        "kitty",
        "lamp",
        "later",
        "like",
        "lion",
        "lips",
        "listen",
        "look",
        "loud",
        "mad",
        "make",
        "man",
        "many",
        "milk",
        "minemy",
        "mitten",
        "mom",
        "moon",
        "morning",
        "mouse",
        "mouth",
        "nap",
        "napkin",
        "night",
        "no",
        "noisy",
        "nose",
        "not",
        "now",
        "nuts",
        "old",
        "on",
        "open",
        "orange",
        "outside",
        "owie",
        "owl",
        "pajamas",
        "pen",
        "pencil",
        "penny",
        "person",
        "pig",
        "pizza",
        "please",
        "police",
        "pool",
        "potty",
        "pretend",
        "pretty",
        "puppy",
        "puzzle",
        "quiet",
        "radio",
        "rain",
        "read",
        "red",
        "refrigerator",
        "ride",
        "room",
        "sad",
        "same",
        "say",
        "scissors",
        "see",
        "shhh",
        "shirt",
        "shoe",
        "shower",
        "sick",
        "sleep",
        "sleepy",
        "smile",
        "snack",
        "snow",
        "stairs",
        "stay",
        "sticky",
        "store",
        "story",
        "stuck",
        "sun",
        "table",
        "talk",
        "taste",
        "thankyou",
        "that",
        "there",
        "think",
        "thirsty",
        "tiger",
        "time",
        "tomorrow",
        "tongue",
        "tooth",
        "toothbrush",
        "touch",
        "toy",
        "tree",
        "uncle",
        "underwear",
        "up",
        "vacuum",
        "wait",
        "wake",
        "water",
        "wet",
        "weus",
        "where",
        "white",
        "who",
        "why",
        "will",
        "wolf",
        "yellow",
        "yes",
        "yesterday",
        "yourself",
        "yucky",
        "zebra",
        "zipper",
    ]
    MAX_FILES_PER_SIGN = 500
    TARGET_FRAMES = 40
    data_cleaner = ASLGraphDataBuilder(
        BASE_DIR, SIGNS_TO_PROCESS, MAX_FILES_PER_SIGN, TARGET_FRAMES
    )
    data_cleaner.process()


if __name__ == "__main__":
    main()
