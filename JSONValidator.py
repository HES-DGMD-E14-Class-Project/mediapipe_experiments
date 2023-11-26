import os
import json
from dotenv import load_dotenv
from collections import defaultdict, Counter

def is_sequential(frames):
    """ Check if frame numbers are sequential """
    return all(frames[i]['frame'] == i for i in range(len(frames)))

def check_frame_counts(output_dir, limit=-1):
    signs_info = defaultdict(lambda: defaultdict(list))
    global_landmarks_counts = []
    frames_with_no_landmarks = 0  # Counter for frames with no landmarks
    non_sequential_files = []  # List to keep track of files with non-sequential frames
    files_processed = 0

    # Iterate over the files in the output directory
    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            if limit != -1 and files_processed >= limit:
                break
            files_processed += 1

            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                sign_label = data['sign']
                file_landmarks_counts = []

                # Iterate over each example and each frame to collect frame and landmark counts
                for example in data['examples']:
                    frame_count = len(example['frames'])
                    signs_info[sign_label]['frame_counts'].append(frame_count)

                    if not is_sequential(example['frames']):
                        non_sequential_files.append(filename)

                    # Collect landmark counts for this file and globally
                    for frame in example['frames']:
                        landmarks_count = len(frame['landmarks'])
                        file_landmarks_counts.append(landmarks_count)
                        global_landmarks_counts.append(landmarks_count)

                        # Check for frames with no landmarks
                        if landmarks_count == 0:
                            frames_with_no_landmarks += 1

                        # Validation for temporal consistency
                        temporal_landmarks = set([temp['landmark'] for temp in frame['temporal']])
                        landmark_types = set(frame['landmark_types'])

                        if len(temporal_landmarks) != len(landmark_types):
                            print(f"Warning: Inconsistent 'temporal' block in frame {frame['frame']} of file {filename}.")
                            print(f"Expected {len(landmark_types)} entries, found {len(temporal_landmarks)}.")

                        if temporal_landmarks != landmark_types:
                            print(f"Warning: 'temporal' block does not match 'landmark_types' in frame {frame['frame']} of file {filename}.")

                signs_info[sign_label]['landmarks_counts'].extend(file_landmarks_counts)

                if file_landmarks_counts:
                    min_landmarks = min(file_landmarks_counts)
                    max_landmarks = max(file_landmarks_counts)
                    avg_landmarks = sum(file_landmarks_counts) / len(file_landmarks_counts)
                    print(f"File '{filename}' stats: Frame Count: {frame_count} ,"
                          f"Landmarks per frame: min={min_landmarks}, max={max_landmarks}, avg={avg_landmarks:.2f}")
                else:
                    print(f"File '{filename}' contains no landmark data.")

    # Process and print the collected information for each sign
    for sign, info in signs_info.items():
        frame_counts = info['frame_counts']
        landmarks_counts = info['landmarks_counts']
        num_examples = len(frame_counts)
        unique_frame_counts = set(frame_counts)

        if landmarks_counts:
            min_landmarks = min(landmarks_counts)
            max_landmarks = max(landmarks_counts)
            avg_landmarks = sum(landmarks_counts) / len(landmarks_counts)
        else:
            min_landmarks = max_landmarks = avg_landmarks = "N/A"

        print(f"{sign}: {num_examples} examples, with {len(unique_frame_counts)} different frame counts => {unique_frame_counts} "
              f"Landmarks per frame: min={min_landmarks}, max={max_landmarks}, avg={avg_landmarks:.2f}")

    # Print global statistics for landmarks across all files
    if global_landmarks_counts:
        global_min_landmarks = min(global_landmarks_counts)
        global_max_landmarks = max(global_landmarks_counts)
        global_avg_landmarks = sum(global_landmarks_counts) / len(global_landmarks_counts)
        print(f"Overall stats for all files: "
              f"Landmarks per frame: min={global_min_landmarks}, max={global_max_landmarks}, avg={global_avg_landmarks:.2f}")
    else:
        print("No landmark data found in any file.")

    # Report if there were any frames with no landmarks
    if frames_with_no_landmarks > 0:
        print(f"There are {frames_with_no_landmarks} frames with no landmarks.")

    # Report if there are files with non-sequential frames
    if non_sequential_files:
        print(f"The following files have non-sequential frame numbers: {', '.join(non_sequential_files)}")
    else:
        print("All files have sequential frame numbers.")

    print(f"Total files processed: {files_processed}")

if __name__ == "__main__":
    load_dotenv()
    BASE_DIR = os.getenv("ASL_SIGNS_BASE_DIRECTORY")
    JSON_DIR = os.path.join(BASE_DIR, f"processed-40-25")
    check_frame_counts(JSON_DIR, limit=-1)
