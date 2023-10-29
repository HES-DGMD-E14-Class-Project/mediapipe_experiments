import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import cv2
import json
import numpy as np
import os
from dotenv import load_dotenv

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


# Define the GUI
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

        # Play Button
        self.play_button = ttk.Button(root, text="Play", command=self.play_landmarks)
        self.play_button.grid(row=3, column=0, columnspan=2)

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

    def play_landmarks(self):
        # Retrieve the selected file from the listbox
        selected_relative_path = self.file_listbox.get(tk.ACTIVE)
        selected_file = self.path_map[selected_relative_path]

        if not selected_file:
            messagebox.showwarning("Warning", "Please select a file.")
            return

        # Read landmarks from the parquet file
        df_landmarks = pd.read_parquet(selected_file)

        # Sort landmarks based on frame and landmark_index
        df_landmarks = df_landmarks.sort_values(by=["frame", "landmark_index"])

        # Create a window to visualize the landmarks
        window_name = "Landmark Visualization"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_width, frame_height = 500, 500

        # Create a unique frame for each unique frame value in df_landmarks
        for frame_number in df_landmarks["frame"].unique():
            frame_landmarks = df_landmarks[df_landmarks["frame"] == frame_number]

            frame = 255 * np.ones(
                (frame_height, frame_width, 3), dtype=np.uint8
            )  # white background

            for _, landmark in frame_landmarks.iterrows():
                if not pd.isna(landmark["x"]) and not pd.isna(landmark["y"]):
                    x = int(landmark["x"] * frame_width)
                    y = int(landmark["y"] * frame_height)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(20) & 0xFF == ord("q"):
                break

        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.geometry("500x500")
    root.mainloop()
