import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats as st


class ParquetFileAnalysis:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.train_df = pd.read_csv(os.path.join(self.base_dir, "train.csv"))

    def analyze(self):
        # 1. Count of All Signs
        unique_signs = self.train_df["sign"].nunique()
        print(f"Total number of unique signs: {unique_signs}")

        # Data structures to store intermediate results
        parquet_file_counts = {}
        frames_counts = {}
        landmarks_per_frame_counts = []

        for _, row in self.train_df.iterrows():
            sign = row["sign"]
            parquet_file_path = os.path.join(self.base_dir, row["path"])

            # 2. Min and Max Number of Associated Parquet Files for Each Sign
            if sign not in parquet_file_counts:
                parquet_file_counts[sign] = 0
            parquet_file_counts[sign] += 1

            # Read parquet file
            parquet_data = pd.read_parquet(parquet_file_path)

            # 3. Min, Max, and Average Number of Frames per Example for Each Sign
            unique_frames = parquet_data["frame"].nunique()
            if sign not in frames_counts:
                frames_counts[sign] = []
            frames_counts[sign].append(unique_frames)

            # 4. Min, Max, and Average Number of Landmarks per Frame
            landmarks_per_frame = parquet_data.groupby("frame").size().tolist()
            landmarks_per_frame_counts.extend(landmarks_per_frame)

        # Display results
        for sign in parquet_file_counts:
            print(f"\nSign: {sign}")
            print(f"Associated Parquet Files: {parquet_file_counts[sign]}")
            print(
                f"Min Frames: {min(frames_counts[sign])}, Max Frames: {max(frames_counts[sign])}, Average Frames: {np.mean(frames_counts[sign]):.2f}, Median Frames: {np.median(frames_counts[sign]):.2f}"
            )
            mode = st.mode(frames_counts[sign])
            print(f"Mode Frames: {mode.mode}, count: {mode.count}")

        print("\nOverall Frame Landmarks Statistics:")
        print(f"Min Landmarks per Frame: {min(landmarks_per_frame_counts)}")
        print(f"Max Landmarks per Frame: {max(landmarks_per_frame_counts)}")
        print(f"Average Landmarks per Frame: {np.mean(landmarks_per_frame_counts):.2f}")
        print(f"Median Landmarks per Frame: {np.median(landmarks_per_frame_counts):.2f}")

        import csv

        header = ["sign", "frames"]

        # Open the file with 'w' mode and pass the file object to csv.DictWriter
        with open('frames_counts.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

            # Iterate over each sign and its frame counts
            for sign, counts in frames_counts.items():
                # You might need to modify how you handle the counts here depending on how you want to record them
                # For example, if you want to record the average frame count for each sign:
                avg_frames = np.mean(counts)
                writer.writerow({'sign': sign, 'frames': avg_frames})

        # Flattening the frames_counts dictionary to get a list of all frame counts
        all_frame_counts = [frame_count for counts in frames_counts.values() for frame_count in counts]

        # Calculating the average frame count across all signs
        average_frame_count = np.mean(all_frame_counts)

        # Printing the average frame count
        print(f"Average frame count across all signs: {average_frame_count:.2f}")

def main():
    load_dotenv()
    BASE_DIR = os.getenv("ASL_SIGNS_BASE_DIRECTORY")

    analyzer = ParquetFileAnalysis(BASE_DIR)
    analyzer.analyze()


if __name__ == "__main__":
    main()
