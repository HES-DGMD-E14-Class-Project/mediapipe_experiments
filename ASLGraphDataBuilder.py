import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from scipy.spatial import distance
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from scipy.signal import savgol_filter
from dotenv import load_dotenv

class ASlGraphDataBuilder:
    def __init__(self, base_dir, signs_to_process, max_files_per_sign):
        self.base_dir = base_dir
        self.signs_to_process = signs_to_process
        self.max_files_per_sign = max_files_per_sign

        # Load the train dataframe and label map
        self.train_df = pd.read_csv(os.path.join(self.base_dir, "train.csv"))
        with open(
            os.path.join(self.base_dir, "sign_to_prediction_index_map.json")
        ) as f:
            self.label_map = json.load(f)

    @staticmethod
    def interpolate_landmarks(df):
        """
        Interpolates missing landmarks based on neighboring frames.
        If interpolation isn't possible (e.g., at the start or end of the sequence),
        use the mean value of the landmark as a fallback.
        """
        for landmark_type in df["type"].unique():
            mask = df["type"] == landmark_type

            # Linear interpolation for missing x and y values
            df.loc[mask, "x"] = df.loc[mask, "x"].interpolate(method="linear", limit_direction="both")
            df.loc[mask, "y"] = df.loc[mask, "y"].interpolate(method="linear", limit_direction="both")

            # If there are still NaNs after interpolation, fill them with mean values
            mean_x = df.loc[mask, "x"].mean()
            mean_y = df.loc[mask, "y"].mean()
            df.loc[mask, "x"] = df.loc[mask, "x"].fillna(mean_x)
            df.loc[mask, "y"] = df.loc[mask, "y"].fillna(mean_y)

        return df

    def drop_z_coordinate(self, df):
        return df.drop(columns=["z"], errors="ignore")

    def normalize_coordinates(self, df):
        centroid = df[["x", "y"]].mean().tolist()
        df["x"] = df["x"] - centroid[0]
        df["y"] = df["y"] - centroid[1]
        return df

    @staticmethod
    def handle_nan_values(df):
        """Handle NaN values via interpolation and dropping."""
        df = ASlGraphDataBuilder.interpolate_landmarks(df)
        # Drop rows with NaN values in the 'x' or 'y' columns
        df.dropna(subset=['x', 'y'], inplace=True)
        return df

    def compute_delta_features(self, df):
        # Calculating deltas per landmark type
        for landmark_type in df["type"].unique():
            mask = df["type"] == landmark_type
            df.loc[mask, "delta_x"] = df.loc[mask, "x"].diff().fillna(0)
            df.loc[mask, "delta_y"] = df.loc[mask, "y"].diff().fillna(0)

        # Calculate wrist acceleration (assuming wrist landmark is at index 0)
        df["acceleration_x"] = df["delta_x"].diff().fillna(0)
        df["acceleration_y"] = df["delta_y"].diff().fillna(0)
        return df

    def calculate_angle(self, point1, point2, point3):
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def format_output(self, df, sign):
        frames = []
        for frame_number, frame_data in df.groupby("frame"):
            frame_info = {
                "frame": int(frame_number),
                "landmarks": [],
                "landmark_types": [],
                "deltas": [],
                "accelerations": [],
            }

            for landmark_type, landmark_data in frame_data.groupby("type"):
                landmarks = landmark_data[["x", "y"]].values.tolist()
                deltas = landmark_data[["delta_x", "delta_y"]].values.tolist()
                accelerations = landmark_data[["acceleration_x", "acceleration_y"]].values.tolist()
                landmark_type = landmark_data["type"].iloc[0]

                frame_info["landmarks"].extend(landmarks)
                frame_info["deltas"].extend(deltas)
                frame_info["accelerations"].extend(accelerations)
                frame_info["landmark_types"].extend([f"{landmark_type}-{int(idx)}" for idx in landmark_data["landmark_index"]])
            frames.append(frame_info)

        result = {"frames": frames}
        return result

    @staticmethod
    def extract_relevant_landmarks(df):
        """
        Extracts the relevant pose, face, right hand, and left hand landmarks from the data.

        Args:
        - df (DataFrame): The data containing the landmarks

        Returns:
        - DataFrame: A dataframe containing the relevant landmarks
        """
        # Define the relevant landmarks to keep
        # Format: 'landmark_type-index'
        relevant_landmarks = [
            *['pose-' + str(i) for i in [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]],
            *['face-' + str(i) for i in [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 61, 62, 63, 64, 66, 291, 292, 293, 294, 295, 296]],
            *['right_hand-' + str(i) for i in range(21)],  # Keeping all right hand landmarks
            *['left_hand-' + str(i) for i in range(21)],   # Keeping all left hand landmarks
        ]

        # Extract landmark type and index from 'row_id'
        df['landmark_type'] = df['row_id'].apply(lambda x: x.split('-')[1])
        df['landmark_index'] = df['row_id'].apply(lambda x: int(x.split('-')[2]))

        # Create a new column 'landmark_id' to help in filtering
        df['landmark_id'] = df['landmark_type'] + '-' + df['landmark_index'].astype(str)

        # Filter the DataFrame based on relevant landmarks
        df_filtered = df[df['landmark_id'].isin(relevant_landmarks)]

        # Retaining the 'landmark_index', 'row_id' and 'frame' columns in the final DataFrame
        df_filtered = df_filtered[['row_id', 'landmark_index', 'x', 'y', 'frame', 'type']]

        # Debugging: Print the DataFrame after filtering based on relevant landmarks
        # print("\nDataFrame After Extracting Relevant Landmarks:")
        # print(df_filtered.head())

        return df_filtered

    def clean_and_save(self):
        all_signs_data = {}

        for sign in tqdm(self.signs_to_process, desc="Processing signs", unit="sign"):
            all_signs_data[sign] = {"sign": sign, "examples": []}
            parquet_files = self._filter_files_by_sign(sign)

            # Step 1: Calculate max_frames for this sign
            max_frames = 0
            for parquet_file in parquet_files:
                df = pd.read_parquet(parquet_file)
                df["sign"] = sign
                num_frames = len(df["frame"].unique())
                if num_frames > max_frames:
                    max_frames = num_frames

            print(f"Max Frames for {sign} ==> {max_frames}")

            for parquet_file in tqdm(parquet_files, desc="Cleaning data", unit="file"):
                df = pd.read_parquet(parquet_file)
                df["sign"] = sign

                # Extract relevant landmarks
                df = self.extract_relevant_landmarks(df)
                df = df.sort_values(by=["frame", "landmark_index"]).reset_index(drop=True)

                # Handle NaN values via interpolation
                df = self.handle_nan_values(df)

                if sign in self.label_map:
                    df["label"] = self.label_map[sign]
                else:
                    print(f"Warning: '{sign}' not found in label map. Skipping...")
                    continue

                df = self.drop_z_coordinate(df)
                df = self.normalize_coordinates(df)
                df = self.compute_delta_features(df)

                # Final NaN check
                if df[["x", "y", "delta_x", "delta_y", "acceleration_x", "acceleration_y"]].isna().any().any():
                    print(f"Warning: NaN values detected for sign '{sign}'. Investigate further.")
                    continue  # Skip processing this sign if NaNs are still present

                df = self.smooth_landmarks(df, window_length=5, polyorder=3)
                df = self.interpolate_frames(df, max_frames)
                example = self.format_output(df, sign)
                all_signs_data[sign]["examples"].append(example)

            output_filename = os.path.join(self.base_dir, f"spatio-temporal/{sign}.json")
            with open(output_filename, "w") as f:
                json.dump(all_signs_data[sign], f, indent=2)

            print(f"Data for sign '{sign}' has been cleaned and saved to {output_filename}")

    def _filter_files_by_sign(self, sign):
        sign_files = self.train_df[self.train_df["sign"] == sign]["path"].tolist()[
            : self.max_files_per_sign
        ]
        return [os.path.join(self.base_dir, f) for f in sign_files]

    def smooth_landmarks(self, df, window_length, polyorder):
        """
        Applies Savitzky-Golay filter to smooth out landmark trajectories.

        :param df: DataFrame, containing landmarks data
        :param window_length: int, the length of the filter window (i.e., the number of coefficients). Must be a positive odd integer.
        :param polyorder: int, the order of the polynomial used to fit the samples. Must be less than `window_length`.
        :return: DataFrame, the same DataFrame with smoothed x and y coordinates
        """
        # Ensure window_length is odd and greater than or equal to polyorder
        if window_length % 2 == 0 or window_length <= polyorder:
            raise ValueError("window_length must be an odd number and >= polyorder.")

        for landmark_type in df["type"].unique():
            mask = df["type"] == landmark_type

            # Ensure that the window length is smaller or equal to the number of frames per landmark type
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

    def interpolate_frames(self, df, max_frames):
        num_frames = len(df["frame"].unique())

        if num_frames >= max_frames:
            return df

        # Ensure that the frame numbers are continuous and start from 0
        df["frame"] = df["frame"].rank(method="dense").astype(int) - 1

        # Calculate the number of frames to interpolate to reach max_frames
        frames_to_add = max_frames - num_frames

        interpolated_frames = []
        for frame_num in range(num_frames - 1):
            current_frame = df[df["frame"] == frame_num].sort_values(by="landmark_index")
            next_frame = df[df["frame"] == frame_num + 1].sort_values(by="landmark_index")

            # Check if the number of landmarks is consistent
            if len(current_frame) != len(next_frame):
                raise ValueError(f"Mismatch in number of landmarks between frames {frame_num} and {frame_num + 1}.")

            # Calculate the difference in landmarks between the current and next frame
            deltas = (next_frame[["x", "y"]].values - current_frame[["x", "y"]].values)

            # Calculate the number of frames to interpolate between the current and next frame
            frames_to_interpolate = frames_to_add // (num_frames - frame_num - 1)
            frames_to_add -= frames_to_interpolate

            # Interpolate frames
            for i in range(frames_to_interpolate + 1):
                interpolated_frame = current_frame.copy()
                interpolated_frame["x"] += deltas[:, 0] * (i / (frames_to_interpolate + 1))
                interpolated_frame["y"] += deltas[:, 1] * (i / (frames_to_interpolate + 1))
                interpolated_frame["frame"] = frame_num * (frames_to_interpolate + 2) + i
                interpolated_frames.append(interpolated_frame)

        # Add the last original frame
        last_frame = df[df["frame"] == num_frames - 1].copy()
        last_frame["frame"] = max_frames - 1
        interpolated_frames.append(last_frame)

        # Concatenate all interpolated frames and sort by frame number
        interpolated_df = pd.concat(interpolated_frames, ignore_index=True).sort_values(by="frame").reset_index(drop=True)

        return interpolated_df

def main():
    load_dotenv()
    BASE_DIR = os.getenv("ASL_SIGNS_BASE_DIRECTORY")
    SIGNS_TO_PROCESS = [
        # "aunt",
        # "after",
        # "all",
        "alligator",
        # "animal",
        # "another",
        # "any",
        # "apple",
        # "airplane",
        # "arm",
        # "awake",
        # "backyard",
        # "bad",
        # "bath",
        # "balloon",
        # "bee",
        "because",
        "bed",
        # "bedroom",
        # "before",
        # "better",
        # "beside",
        # "bird",
        # "black",
        # "blue",
        # "blow",
        # "boat",
        # "book",
        "boy",
        # "brother",
        # "brown",
        # "bug",
        # "bye",
        "callonphone",
        # "can",
        # "car",
        # "carrot",
        # "cat",
        # "cereal",
        # "chair",
        # "cheek",
        "child",
        # "chin",
        # "chocolate",
        "clown",
        # "closet",
        # "cloud",
        # "cow",
        # "cowboy",
        # "cute",
        # "cut",
        # "cry",
        # "dance",
         "dad",
        "dog",
        # "doll",
        # "donkey",
        # "down",
        # "drawer",
        # "drink",
        # "dry",
        # "dryer",
        # "duck",
        # "ear",
        # "elephant",
        # "empty",
        # "every",
        # "eye",
         "face",
         "fall",
        # "farm",
        # "fast",
        # "feet",
        # "finger",
        # "finish",
        # "fireman",
        # "fish",
        # "flag",
        # "flower",
        # "food",
        # "for",
        # "frog",
        # "fine",
        # "find",
        # "first",
        # "frenchfries",
        # "garbage",
        # "gift",
        # "girl",
         "giraffe",
        # "glasswindow",
        # "go",
        # "goose",
        # "green",
        # "gum",
        # "happy",
        # "hair",
        # "have",
        # "haveto",
        # "hate",
        # "hat",
        # "head",
        # "hear",
         "hello",
        # "hen",
        # "hesheit",
        # "helicopter",
        # "high",
        # "hide",
        # "horse",
        # "hot",
        # "hungry",
        # "icecream",
        # "if",
        # "into",
         "jump",
        # "jacket",
        # "jeans",
        # "kiss",
        # "kitty",
        # "lamp",
        # "later",
        # "like",
        # "listen",
        # "lion",
        # "lips",
        # "look",
        # "loud",
        # "mad",
        # "make",
        # "man",
        # "many",
        # "milk",
        # "minemy",
        # "mitten",
        # "mom",
         "moon",
         "morning",
        # "mouse",
        # "mouth",
        # "nap",
        # "napkin",
        # "no",
        # "noisy",
        # "not",
        # "now",
        # "nuts",
        # "old",
        # "on",
        # "open",
        # "orange",
        # "outside",
        # "owl",
        # "pajamas",
        # "penny",
        # "pencil",
        # "person",
        # "pizza",
        "please",
        # "police",
        # "pool",
        # "potty",
        # "pretend",
        # "pretty",
        # "puppy",
        # "puzzle",
        # "quiet",
        # "rain",
        # "radio",
        # "read",
        # "red",
        # "refrigerator",
        # "ride",
        # "room",
        # "say",
        # "scissors",
        # "see",
        # "shhh",
        # "shirt",
        # "shoe",
        # "shower",
        # "sick",
        # "sleep",
        # "sleepy",
        # "smile",
        # "snow",
        # "snack",
        # "sticky",
        # "stay",
        # "store",
        # "sun",
        # "talk",
        # "taste",
        # "table",
        # "that",
         "thankyou",
        # "there",
        # "think",
        # "thirsty",
        # "tiger",
        # "time",
        # "tooth",
        # "toothbrush",
        # "tongue",
        # "touch",
        # "toy",
        # "tree",
        # "TV",
        # "up",
        # "uncle",
        # "underwear",
        # "vacuum",
         "wait",
        # "wake",
        # "water",
        # "weus",
        # "wet",
        # "where",
        # "white",
        # "who",
        # "why",
        # "will",
        # "wolf",
        # "yellow",
         "yes",
        # "yesterday",
        # "yucky",
        # "zebra",
    ]
    MAX_FILES_PER_SIGN = 100

    data_cleaner = ASlGraphDataBuilder(BASE_DIR, SIGNS_TO_PROCESS, MAX_FILES_PER_SIGN)
    data_cleaner.clean_and_save()


if __name__ == "__main__":
    main()
