import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from scipy.spatial import distance
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from scipy.signal import savgol_filter
from dotenv import load_dotenv


class ASLDataCleaner:
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

    def drop_z_coordinate(self, df):
        return df.drop(columns=["z"], errors="ignore")

    def normalize_coordinates(self, df):
        for part_type in df["type"].unique():
            mask = df["type"] == part_type
            min_val = df.loc[mask, ["x", "y"]].min()
            range_val = df.loc[mask, ["x", "y"]].max() - min_val

            # Check for 0 range to avoid division by 0
            range_val = np.where(range_val == 0, 1, range_val)

            df.loc[mask, ["x", "y"]] = (df.loc[mask, ["x", "y"]] - min_val) / range_val
        return df

    def handle_nan_values(self, df):
        # Option 1: Replace NaN values
        df.fillna(0, inplace=True)

        # Option 2: Remove rows with NaN values
        # df = df.dropna()

        return df

    def compute_delta_features(self, df):
        """
        Compute the differences (delta_x and delta_y) for each landmark type separately.
        This ensures that the delta for the first landmark of type "right_hand" in frame n+1
        is computed relative to the same landmark in frame n, rather than the last landmark
        of type "pose" in frame n
        """
        # Ensuring the data is sorted by frame
        df = df.sort_values(by=["type", "landmark_index", "frame"])

        # Calculating deltas per landmark type
        for landmark_type in df["type"].unique():
            mask = df["type"] == landmark_type
            df.loc[mask, "delta_x"] = df.loc[mask, "x"].diff().fillna(0)
            df.loc[mask, "delta_y"] = df.loc[mask, "y"].diff().fillna(0)

            # Ensuring that deltas between different videos/signs are not calculated
            df.loc[mask & (df["frame"].diff() != 1), ["delta_x", "delta_y"]] = 0

        return df

    def format_output(self, df, sign):
        frames = []
        type_mapping = {"face": "F", "pose": "P", "right_hand": "R", "left_hand": "L"}

        for _, frame_data in df.groupby("frame"):
            frame_info = {
                "frame": int(frame_data["frame"].iloc[0]),
                "sign": sign,
                "landmarks": frame_data[["x", "y"]].values.tolist(),
                "landmark_types": frame_data["type"].map(type_mapping).tolist(),
                "deltas": frame_data[["delta_x", "delta_y"]].values.tolist(),
            }
            frames.append(frame_info)
        return {"sign": sign, "frames": frames}

    def clean_and_save(self):
        for sign in self.signs_to_process:
            # Get parquet files for the current sign
            parquet_files = self._filter_files_by_sign(sign)

            # Process the files and produce a dataframe
            df = self._read_and_concatenate_parquet_files(parquet_files, sign)

            df["label"] = self.label_map[sign]

            # Handle NaN values
            df = self.handle_nan_values(df)

            # Data Cleaning Steps
            df = self.drop_z_coordinate(df)
            df = self.normalize_coordinates(df)
            df = self.compute_delta_features(df)
            df = self.remove_outliers(df)
            df = self.remove_trajectory_outliers(df)
            df = self.smooth_landmarks(df, window_length=5, polyorder=3)
            df = self.interpolate_missing_landmarks(df)
            df = self.interpolate_frames(df)

            # Format and Save Output
            output_json = self.format_output(df, sign)
            output_filename = os.path.join(self.base_dir, f"cleaned/{sign}.json")
            with open(output_filename, "w") as f:
                json.dump(output_json, f)
            print(
                f"Data for sign '{sign}' has been cleaned and saved to {output_filename}"
            )

    def _filter_files_by_sign(self, sign):
        sign_files = self.train_df[self.train_df["sign"] == sign]["path"].tolist()[
            : self.max_files_per_sign
        ]
        return [os.path.join(self.base_dir, f) for f in sign_files]

    def _read_and_concatenate_parquet_files(self, parquet_files, sign):
        dfs = []
        for f in tqdm(parquet_files, desc="Reading Parquet Files"):
            relative_path = os.path.relpath(f, self.base_dir)
            matched_rows = self.train_df[self.train_df["path"] == relative_path]
            if matched_rows.empty:
                print(f"Warning: No match found for file {f}")
                continue
            df = pd.read_parquet(f)
            df["sign"] = sign  # Add sign information here
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def remove_outliers(self, df):
        """
        Z-Score Based Outlier Removal:
        We can use the Z-score to detect outliers in the landmarks' coordinates.
        The Z-score represents how many standard deviations a data point is from the mean.
        Points with a Z-score above a certain threshold can be considered outliers.
        """
        # Compute Z-scores of ‘x’ and ‘y’ coordinates across all landmarks and frames.
        df[["z_x", "z_y"]] = df.groupby(["frame"])[["x", "y"]].transform(
            lambda g: zscore(g, ddof=1)
        )

        # Define a threshold above which a point will be considered an outlier.
        z_threshold = 3  # example threshold

        # Identify outliers.
        outliers = np.abs(df[["z_x", "z_y"]]) > z_threshold

        # Cap the Outliers.
        # we cap the outliers to the mean ± 3 standard deviations,
        # retaining the original data structure but reducing the impact of extreme values.
        for coord in ["x", "y"]:
            lower_bound = df.groupby(["frame"])[coord].transform(
                lambda g: g.mean() - z_threshold * g.std()
            )
            upper_bound = df.groupby(["frame"])[coord].transform(
                lambda g: g.mean() + z_threshold * g.std()
            )

            df[coord] = np.where(df["z_" + coord] > z_threshold, upper_bound, df[coord])
            df[coord] = np.where(
                df["z_" + coord] < -z_threshold, lower_bound, df[coord]
            )

        return df.drop(columns=["z_x", "z_y"])

    def remove_trajectory_outliers(self, df):
        """
        Trajectory-based Outlier Detection:
        In the case of video data, you may also want to consider temporal aspects,
        ensuring the landmarks’ trajectories are smooth and physically plausible.
        Anomaly detection algorithms like Isolation Forest, DBSCAN, or specialized
        trajectory analysis can be applied.
        """
        # Train an Isolation Forest on the deltas of x and y coordinates.
        df["delta_x"] = df.groupby(["row_id"])["x"].diff().fillna(0)
        df["delta_y"] = df.groupby(["row_id"])["y"].diff().fillna(0)

        model = IsolationForest(contamination=0.01)  # 1% of data is outliers
        outliers = model.fit_predict(df[["delta_x", "delta_y"]])

        # Remove identified outlier frames.
        df_cleaned = df[outliers != -1]

        return df_cleaned

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

    def interpolate_missing_landmarks(self, df):
        """
        Interpolate missing or unreliable landmarks based on neighboring frames.

        Parameters:
            df (pd.DataFrame): Dataframe containing landmark data.

        Returns:
            pd.DataFrame: Dataframe with interpolated landmarks.
        """
        # Assuming df is a DataFrame where each row represents a landmark in a specific frame
        # and contains at least columns ['frame', 'x', 'y'] (and maybe more columns for other coordinates, types, etc.)

        for landmark_type in df["type"].unique():
            mask = df["type"] == landmark_type

            # Linear interpolation for missing x and y values
            df.loc[mask, "x"] = df.loc[mask, "x"].interpolate(method="linear")
            df.loc[mask, "y"] = df.loc[mask, "y"].interpolate(method="linear")

        # Handle remaining NaN values that could not be interpolated (e.g., at the start or end of a sequence)
        df = df.dropna(subset=["x", "y"])

        return df

    def interpolate_frames(self, df, desired_length=None):
        """
        Interpolate frames to reach the desired sequence length.

        Parameters:
            df (pd.DataFrame): Dataframe containing landmark data.
            desired_length (int, optional): The target number of frames per video.
                If None, the median length of videos is used.

        Returns:
            pd.DataFrame: Dataframe with interpolated frames.
        """
        if desired_length is None:
            desired_length = int(df.groupby("sign")["frame"].nunique().median())

        # Creating new frame indices
        new_frames = np.linspace(df["frame"].min(), df["frame"].max(), desired_length)

        # Creating a new DataFrame with interpolated landmarks for the new frames
        new_df = pd.DataFrame()
        for landmark_type in df["type"].unique():
            mask = df["type"] == landmark_type
            # Only keep numeric columns when computing the mean
            numeric_df = df[mask].select_dtypes(include=[np.number])
            temp_df = numeric_df.groupby("frame").mean()

            # Interpolating x and y values
            temp_df = temp_df.reindex(new_frames).interpolate(method="linear")
            temp_df["type"] = landmark_type
            new_df = pd.concat([new_df, temp_df])

        new_df.reset_index(inplace=True)
        new_df.rename(columns={"index": "frame"}, inplace=True)

        return new_df


def main():
    load_dotenv()
    BASE_DIR = os.getenv("ASL_SIGNS_BASE_DIRECTORY")
    SIGNS_TO_PROCESS = [
        "aunt",
        "after",
        "all",
        "alligator",
        "animal",
        "another",
        "any",
        "apple",
        "airplane",
        "arm",
        "awake",
        "backyard",
        "bad",
        "bath",
        "balloon",
        "bee",
        "because",
        "bed",
        "bedroom",
        "before",
        "better",
        "beside",
        "bird",
        "black",
        "blue",
        "blow",
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
        "clown",
        "closet",
        "cloud",
        "cow",
        "cowboy",
        "cute",
        "cut",
        "cry",
        "dance",
        "dad",
        "dog",
        "doll",
        "donkey",
        "down",
        "drawer",
        "drink",
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
        "finger",
        "finish",
        "fireman",
        "fish",
        "flag",
        "flower",
        "food",
        "for",
        "frog",
        "fine",
        "find",
        "first",
        "frenchfries",
        "garbage",
        "gift",
        "girl",
        "giraffe",
        "glasswindow",
        "go",
        "goose",
        "green",
        "gum",
        "happy",
        "hair",
        "have",
        "haveto",
        "hate",
        "hat",
        "head",
        "hear",
        "hello",
        "hen",
        "hesheit",
        "helicopter",
        "high",
        "hide",
        "horse",
        "hot",
        "hungry",
        "icecream",
        "if",
        "into",
        "interpolate",
        "jump",
        "jacket",
        "jeans",
        "kiss",
        "kitty",
        "lamp",
        "later",
        "like",
        "listen",
        "lion",
        "lips",
        "look",
        "loud",
        "love",
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
        "no",
        "noisy",
        "not",
        "now",
        "nuts",
        "old",
        "on",
        "open",
        "orange",
        "outside",
        "owl",
        "pajamas",
        "penny",
        "pencil",
        "person",
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
        "rain",
        "radio",
        "read",
        "red",
        "refrigerator",
        "ride",
        "room",
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
        "snow",
        "snack",
        "sticky",
        "stay",
        "store",
        "sun",
        "talk",
        "taste",
        "table",
        "that",
        "thankyou",
        "there",
        "think",
        "thirsty",
        "tiger",
        "time",
        "tooth",
        "toothbrush",
        "tongue",
        "touch",
        "toy",
        "tree",
        "TV",
        "up",
        "uncle",
        "underwear",
        "vacuum",
        "wait",
        "wake",
        "water",
        "weus",
        "wet",
        "where",
        "white",
        "who",
        "why",
        "will",
        "wolf",
        "yellow",
        "yes",
        "yesterday",
        "yucky",
        "zebra",
    ]
    MAX_FILES_PER_SIGN = 400

    data_cleaner = ASLDataCleaner(BASE_DIR, SIGNS_TO_PROCESS, MAX_FILES_PER_SIGN)
    data_cleaner.clean_and_save()


if __name__ == "__main__":
    main()
