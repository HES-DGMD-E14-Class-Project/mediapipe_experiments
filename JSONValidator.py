import os
import json
from dotenv import load_dotenv
from collections import defaultdict, Counter

def check_frame_counts(output_dir):
    signs_info = defaultdict(list)

    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                sign_label = data['sign']
                frame_counts = [len(example['frames']) for example in data['examples']]
                signs_info[sign_label].extend(frame_counts)

    for sign, frame_counts in signs_info.items():
        num_examples = len(frame_counts)
        unique_frame_counts = set(frame_counts)

        if len(unique_frame_counts) == 1:
            print(f"{sign}: {num_examples} examples, all with {unique_frame_counts.pop()} frames.")
        else:
            frame_count_distribution = Counter(frame_counts)
            mismatched_info = ", ".join([f"{count} examples with {frames} frames" for frames, count in frame_count_distribution.items()])
            print(f"{sign}: {num_examples} examples, with discrepancies in frame counts. {mismatched_info}")

if __name__ == "__main__":
    load_dotenv()
    BASE_DIR = os.getenv("ASL_SIGNS_BASE_DIRECTORY")
    JSON_DIR = os.path.join(BASE_DIR, f"spatio-temporal")
    check_frame_counts(JSON_DIR)
