import pandas as pd

# Step 1: Load the Parquet File
file_path = "/Users/brian.sam-bodden/Documents/hes/DGMD_E-14/Project/Datasets/asl-signs/train_landmark_files/62590/1002885072.parquet"
df = pd.read_parquet(file_path)

# # Step 2: Check the Unique Landmarks and Frames
# unique_landmarks = df['landmark_index'].unique()
# unique_frames = df['frame'].unique()
# landmarks_per_frame = df.groupby('frame').size()

# # Step 3: Print Debug Information
# print(f"Unique Landmarks: {unique_landmarks}")
# print(f"Number of Unique Frames: {len(unique_frames)}")
# print(f"Landmarks per Frame:\n{landmarks_per_frame}")

# Step 1: Verify Landmark Types
unique_types = df['type'].unique()
print(f"Unique Landmark Types: {unique_types}")

# Step 2: Check Landmarks per Type
landmarks_per_type_and_frame = df.groupby(['frame', 'type']).size().unstack(fill_value=0)
print(f"Landmarks per Type and Frame:\n{landmarks_per_type_and_frame}")

# Step 3: Inspect Specific Frames (Optional)
# If you find any inconsistencies, you might want to inspect specific frames.
# For example, to inspect frame 96:
frame_96_data = df[df['frame'] == 96]
print(f"Data for Frame 96:\n{frame_96_data}")
