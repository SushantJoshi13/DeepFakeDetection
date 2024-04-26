import json
import glob
import numpy as np
import cv2
import os
import face_recognition
from tqdm import tqdm

# Modify the path to your directory containing the videos
video_directory = '/Users/avikshitkharkar/Documents/deepfake-detection-challenge/dataset/'

# Use glob to find all .mp4 files in the directory
video_files = glob.glob(video_directory + '*.mp4')

def frame_extract(path):
    vidObj = cv2.VideoCapture(path) 
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image

def create_face_videos(path_list, out_dir):
    already_present_count = glob.glob(os.path.join(out_dir, '*.mp4'))
    print("Number of videos already present:", len(already_present_count))
    
    for path in tqdm(path_list):
        out_path = os.path.join(out_dir, os.path.basename(path))
        if os.path.exists(out_path):
            print("File already exists:", out_path)
            continue
        
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), 30, (112, 112))
        
        for idx, frame in enumerate(frame_extract(path)):
            if idx <= 90:  # Process only first 150 frames
                faces = face_recognition.face_locations(frame)
                for face in faces:
                    top, right, bottom, left = face
                    try:
                        out.write(cv2.resize(frame[top:bottom, left:right, :], (112, 112)))
                    except:
                        pass
        out.release()

# Modify the output directory path to your desired directory
output_directory = '/Users/avikshitkharkar/Documents/deepfake-detection-challenge/Real_faces'
create_face_videos(video_files, output_directory)
