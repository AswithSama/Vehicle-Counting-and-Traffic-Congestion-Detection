import tkinter as tk
from tkinter import filedialog

def get_video_path():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
    )
    return video_path
