import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import json
import numpy as np
from dotenv import load_dotenv
import cv2
from PIL import Image, ImageTk

# Load environment variables / secrets from .env file.
load_dotenv()

BASE_DIRECTORY = os.getenv("ASL_SIGNS_BASE_DIRECTORY")

# Read the list of labels from JSON file
with open(f"{BASE_DIRECTORY}/sign_to_prediction_index_map.json", "r") as f:
    data = json.load(f)
    labels = list(data.keys())

# Read the train.csv file
df = pd.read_csv(f"{BASE_DIRECTORY}/train.csv")
print(df.columns)


class VideoPlayer(tk.Toplevel):
    HAND_CONNECTIONS = frozenset(
        [
            # Palm
            (0, 1),
            (0, 5),
            (9, 13),
            (13, 17),
            (5, 9),
            (0, 17),
            # Thumb
            (1, 2),
            (2, 3),
            (3, 4),
            # Index finger
            (5, 6),
            (6, 7),
            (7, 8),
            # Middle finger
            (9, 10),
            (10, 11),
            (11, 12),
            # Ring finger
            (13, 14),
            (14, 15),
            (15, 16),
            # Pinky
            (17, 18),
            (18, 19),
            (19, 20),
        ]
    )

    POSE_CONNECTIONS = frozenset(
        [
            # Updated pose connections excluding hands and face
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 7),
            (0, 4),
            (4, 5),
            (5, 6),
            (6, 8),
            (9, 10),
            (11, 12),
            (11, 13),
            (13, 15),
            (15, 17),
            (12, 14),
            (14, 16),
            (16, 18),
            (11, 23),
            (12, 24),
            (23, 24),
        ]
    )

    # FACEMESH

    FACEMESH_LIPS = frozenset(
        [
            (61, 146),
            (146, 91),
            (91, 181),
            (181, 84),
            (84, 17),
            (17, 314),
            (314, 405),
            (405, 321),
            (321, 375),
            (375, 291),
            (61, 185),
            (185, 40),
            (40, 39),
            (39, 37),
            (37, 0),
            (0, 267),
            (267, 269),
            (269, 270),
            (270, 409),
            (409, 291),
            (78, 95),
            (95, 88),
            (88, 178),
            (178, 87),
            (87, 14),
            (14, 317),
            (317, 402),
            (402, 318),
            (318, 324),
            (324, 308),
            (78, 191),
            (191, 80),
            (80, 81),
            (81, 82),
            (82, 13),
            (13, 312),
            (312, 311),
            (311, 310),
            (310, 415),
            (415, 308),
        ]
    )

    FACEMESH_LEFT_EYE = frozenset(
        [
            (263, 249),
            (249, 390),
            (390, 373),
            (373, 374),
            (374, 380),
            (380, 381),
            (381, 382),
            (382, 362),
            (263, 466),
            (466, 388),
            (388, 387),
            (387, 386),
            (386, 385),
            (385, 384),
            (384, 398),
            (398, 362),
        ]
    )

    FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477), (477, 474)])

    FACEMESH_LEFT_EYEBROW = frozenset(
        [
            (276, 283),
            (283, 282),
            (282, 295),
            (295, 285),
            (300, 293),
            (293, 334),
            (334, 296),
            (296, 336),
        ]
    )

    FACEMESH_RIGHT_EYE = frozenset(
        [
            (33, 7),
            (7, 163),
            (163, 144),
            (144, 145),
            (145, 153),
            (153, 154),
            (154, 155),
            (155, 133),
            (33, 246),
            (246, 161),
            (161, 160),
            (160, 159),
            (159, 158),
            (158, 157),
            (157, 173),
            (173, 133),
        ]
    )

    FACEMESH_RIGHT_EYEBROW = frozenset(
        [
            (46, 53),
            (53, 52),
            (52, 65),
            (65, 55),
            (70, 63),
            (63, 105),
            (105, 66),
            (66, 107),
        ]
    )

    FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472), (472, 469)])

    FACEMESH_FACE_OVAL = frozenset(
        [
            (10, 338),
            (338, 297),
            (297, 332),
            (332, 284),
            (284, 251),
            (251, 389),
            (389, 356),
            (356, 454),
            (454, 323),
            (323, 361),
            (361, 288),
            (288, 397),
            (397, 365),
            (365, 379),
            (379, 378),
            (378, 400),
            (400, 377),
            (377, 152),
            (152, 148),
            (148, 176),
            (176, 149),
            (149, 150),
            (150, 136),
            (136, 172),
            (172, 58),
            (58, 132),
            (132, 93),
            (93, 234),
            (234, 127),
            (127, 162),
            (162, 21),
            (21, 54),
            (54, 103),
            (103, 67),
            (67, 109),
            (109, 10),
        ]
    )

    FACEMESH_NOSE = frozenset(
        [
            (168, 6),
            (6, 197),
            (197, 195),
            (195, 5),
            (5, 4),
            (4, 1),
            (1, 19),
            (19, 94),
            (94, 2),
            (98, 97),
            (97, 2),
            (2, 326),
            (326, 327),
            (327, 294),
            (294, 278),
            (278, 344),
            (344, 440),
            (440, 275),
            (275, 4),
            (4, 45),
            (45, 220),
            (220, 115),
            (115, 48),
            (48, 64),
            (64, 98),
        ]
    )

    FACEMESH_CONTOURS = frozenset().union(
        *[
            FACEMESH_LIPS,
            FACEMESH_LEFT_EYE,
            FACEMESH_LEFT_EYEBROW,
            FACEMESH_RIGHT_EYE,
            FACEMESH_RIGHT_EYEBROW,
            FACEMESH_FACE_OVAL,
        ]
    )

    FACEMESH_IRISES = frozenset().union(*[FACEMESH_LEFT_IRIS, FACEMESH_RIGHT_IRIS])

    FACE_CONNECTIONS = frozenset().union(
        FACEMESH_LIPS,
        FACEMESH_LEFT_EYE,
        FACEMESH_RIGHT_EYE,
        FACEMESH_FACE_OVAL,
        FACEMESH_LEFT_EYEBROW,
        FACEMESH_RIGHT_EYEBROW
        # If you want to include the irises, add FACEMESH_IRISES
    )

    def __init__(self, master, df_landmarks, file_path, total_frames):
        super().__init__(master)

        # Extracting the filename
        filename = os.path.basename(file_path)

        # Extracting the immediate directory
        directory = os.path.basename(os.path.dirname(file_path))

        self.title(f"Video Player :: {directory}/{filename}")
        self.geometry("500x600")
        self.file_path = file_path
        self.total_frames = total_frames
        self.frame_number = 0
        self.playing = False
        self.df_landmarks = df_landmarks
        # pd.read_parquet(self.file_path)
        self.df_landmarks = self.df_landmarks.sort_values(
            by=["frame", "landmark_index"]
        )

        # Create a mapping from dataset frame numbers to a 0-based index
        frame_numbers = self.df_landmarks['frame'].unique()
        self.frame_mapping = {frame_num: index for index, frame_num in enumerate(frame_numbers)}

        # Total frames is the count of unique frames
        self.total_frames = len(frame_numbers)

        # Play/Pause Button
        self.play_var = tk.StringVar(value="▶️ Play")
        self.play_button = ttk.Button(
            self, textvariable=self.play_var, command=self.toggle_play
        )
        self.play_button.pack(side="top", fill="x")

        # Video Frame
        self.video_frame = ttk.Label(self)
        self.video_frame.pack(side="top", fill="both", expand=True)

        # Frame Label
        self.frame_label = ttk.Label(
            self, text=f"Frame {self.frame_number + 1}/{self.total_frames}"
        )
        self.frame_label.pack(side="top", fill="x")

        # Frame Slider
        self.frame_slider = ttk.Scale(
            self,
            from_=0,
            to=self.total_frames,
            orient="horizontal",
            command=self.update_frame,
        )
        self.frame_slider.pack(side="bottom", fill="x")

        # Add a checkbox for showing connections
        self.show_connections_var = tk.BooleanVar(value=False)
        self.show_connections_checkbox = ttk.Checkbutton(
            self, text="Connected", variable=self.show_connections_var
        )
        self.show_connections_checkbox.pack(side="top", fill="x")

        self.display_frame()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_var.set("⏸️ Pause")
            self.frame_slider.config(state="readonly")
            self.play_frames()
        else:
            self.play_var.set("▶️ Play")
            self.frame_slider.config(state="normal")

    def play_frames(self):
        if not self.playing:
            return

        self.display_frame()
        self.frame_number = (self.frame_number + 1) % self.total_frames
        self.frame_slider.set(self.frame_number)

        if self.playing:
            self.after(33, self.play_frames)  # Adjusted delay for 30 fps

    def draw_hand_connections(self, frame, hand_landmarks, frame_width, frame_height):
        for start, end in self.HAND_CONNECTIONS:
            if start < len(hand_landmarks) and end < len(hand_landmarks):
                start_landmark = hand_landmarks.iloc[start]
                end_landmark = hand_landmarks.iloc[end]
                if (
                    not pd.isna(start_landmark["x"])
                    and not pd.isna(start_landmark["y"])
                    and not pd.isna(end_landmark["x"])
                    and not pd.isna(end_landmark["y"])
                ):
                    start_point = (
                        int(start_landmark["x"] * frame_width),
                        int(start_landmark["y"] * frame_height),
                    )
                    end_point = (
                        int(end_landmark["x"] * frame_width),
                        int(end_landmark["y"] * frame_height),
                    )
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 1)

    def draw_pose_connections(self, frame, pose_landmarks, frame_width, frame_height):
        # Define the indices for the face landmarks in the pose model that you want to exclude
        face_landmarks_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}  # Add the correct indices here

        for start, end in self.POSE_CONNECTIONS:
            # Skip connections if either end is a face landmark index
            if start in face_landmarks_indices or end in face_landmarks_indices:
                continue

            if start < len(pose_landmarks) and end < len(pose_landmarks):
                start_landmark = pose_landmarks.iloc[start]
                end_landmark = pose_landmarks.iloc[end]
                if (
                    not pd.isna(start_landmark["x"])
                    and not pd.isna(start_landmark["y"])
                    and not pd.isna(end_landmark["x"])
                    and not pd.isna(end_landmark["y"])
                ):
                    start_point = (
                        int(start_landmark["x"] * frame_width),
                        int(start_landmark["y"] * frame_height),
                    )
                    end_point = (
                        int(end_landmark["x"] * frame_width),
                        int(end_landmark["y"] * frame_height),
                    )
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 1)

    def draw_face_connections(self, frame, face_landmarks, frame_width, frame_height):
        for start, end in self.FACE_CONNECTIONS:
            if start < len(face_landmarks) and end < len(face_landmarks):
                start_landmark = face_landmarks.iloc[start]
                end_landmark = face_landmarks.iloc[end]
                if (
                    not pd.isna(start_landmark["x"])
                    and not pd.isna(start_landmark["y"])
                    and not pd.isna(end_landmark["x"])
                    and not pd.isna(end_landmark["y"])
                ):
                    start_point = (
                        int(start_landmark["x"] * frame_width),
                        int(start_landmark["y"] * frame_height),
                    )
                    end_point = (
                        int(end_landmark["x"] * frame_width),
                        int(end_landmark["y"] * frame_height),
                    )
                    cv2.line(
                        frame, start_point, end_point, (0, 255, 0), 1
                    )  # Color for face connections

    def display_frame(self):
        frame_width, frame_height = 500, 500

        # Use the mapping to get the actual frame number from the dataset
        dataset_frame_number = list(self.frame_mapping.keys())[self.frame_number]

        frame_landmarks = self.df_landmarks[self.df_landmarks["frame"] == dataset_frame_number]

        frame = 255 * np.ones(
            (frame_height, frame_width, 3), dtype=np.uint8
        )  # white background

        for _, landmark in frame_landmarks.iterrows():
            if not pd.isna(landmark["x"]) and not pd.isna(landmark["y"]):
                x = int(landmark["x"] * frame_width)
                y = int(landmark["y"] * frame_height)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Draw connections for hand landmarks if checkbox is checked
        if self.show_connections_var.get():
            for hand_type in ["left_hand", "right_hand"]:
                hand_landmarks = frame_landmarks[frame_landmarks["type"] == hand_type]

                if hand_landmarks.isnull().all().all():
                    # print(f"{hand_type} is not visible in this frame.")
                    continue

                if self.show_connections_var.get():
                    self.draw_hand_connections(
                        frame, hand_landmarks, frame_width, frame_height
                    )

            pose_landmarks = frame_landmarks[frame_landmarks["type"] == "pose"]
            if not pose_landmarks.empty:
                self.draw_pose_connections(
                    frame, pose_landmarks, frame_width, frame_height
                )

            # Draw connections for face landmarks if checkbox is checked
            face_landmarks = frame_landmarks[frame_landmarks["type"] == "face"]
            if not face_landmarks.empty:
                self.draw_face_connections(
                    frame, face_landmarks, frame_width, frame_height
                )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_width, frame_height))
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))

        self.video_frame.config(image=photo)
        self.video_frame.image = photo

        # Update the frame label to show both player position and actual frame number
        self.frame_label.config(
            text=f"{self.frame_number + 1}/{self.total_frames} - frame: {dataset_frame_number}"
        )

    def update_frame(self, value):
        self.frame_number = int(float(value))
        self.display_frame()


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualize Dataset")

        # Drop-down for Sign
        self.label_var = tk.StringVar()
        self.label_dropdown = ttk.Combobox(root, textvariable=self.label_var)
        self.label_dropdown["values"] = labels
        self.label_dropdown.bind("<<ComboboxSelected>>", self.on_dropdown_select)
        ttk.Label(root, text="Sign").grid(row=0, column=0, sticky="w")
        self.label_dropdown.grid(row=0, column=1, sticky="ew")

        # Label for Number of Parquet Files
        self.num_files_label = ttk.Label(root, text="Number of Examples: 0")
        self.num_files_label.grid(row=1, column=0, columnspan=2, sticky="w")

        # Listbox for files
        self.file_listbox = tk.Listbox(root, height=10, width=40)
        self.file_listbox.grid(row=2, column=0, columnspan=2, sticky="nsew")

        # Open Button
        self.open_button = ttk.Button(root, text="Open", command=self.open_video_player)
        self.open_button.grid(row=3, column=0, columnspan=2)

        # Configure weights
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=1)

    def on_dropdown_select(self, event):
        # Clear the listbox
        self.file_listbox.delete(0, tk.END)

        # Fetch the selected sign from the dropdown
        selected_label = self.label_var.get()

        # Get relative paths from dataframe for the selected label
        relative_files = df[df["sign"] == selected_label]["path"].tolist()

        # Convert relative paths to full paths
        full_paths = [os.path.join(BASE_DIRECTORY, path) for path in relative_files]

        # Dictionary to map relative paths to full paths
        self.path_map = dict(zip(relative_files, full_paths))

        # Update label for number of parquet files
        num_files = len(relative_files)
        self.num_files_label.config(text=f"Number of Examples: {num_files}")

        # Populate the listbox with relative paths
        for relative_path in relative_files:
            self.file_listbox.insert(tk.END, relative_path)

    def open_video_player(self):
        selected_relative_path = self.file_listbox.get(tk.ACTIVE)
        if not selected_relative_path:
            messagebox.showwarning("Warning", "Please select a file.")
            return

        selected_file = self.path_map[selected_relative_path]
        df_landmarks = pd.read_parquet(selected_file)
        total_frames = df_landmarks["frame"].nunique()

        VideoPlayer(self.root, df_landmarks, selected_file, total_frames)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.geometry("500x500")
    root.mainloop()
