import mediapipe as mp
import cv2
import numpy as np
import os
from tqdm import tqdm

mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def feature_extractor_video(video_path, key_points_path):
    cap = cv2.VideoCapture(video_path)
    # extract video name
    video_name = video_path.split('/')[-1].split('.')[0]
    # create vid folder
    vid_folder = os.path.join(key_points_path, video_name)
    if not os.path.exists(vid_folder):
        os.makedirs(vid_folder)

    pose_kp = []
    face_kp = []
    left_hand_kp = []
    right_hand_kp = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False,
    model_complexity=2, enable_segmentation=False) as holistic:
        idx = 0
        # Loop through video
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # extract keypoints
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros(shape=(33,4))
            pose_kp.append(pose)

            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros(shape=(468,3))
            face_kp.append(face)

            left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros(shape=(21,3))
            left_hand_kp.append(left_hand)

            right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros(shape=(21,3))
            right_hand_kp.append(right_hand)

            idx += 1
    
    # save keypoints
    pose_kp = np.array(pose_kp)
    face_kp = np.array(face_kp)
    left_hand_kp = np.array(left_hand_kp)
    right_hand_kp = np.array(right_hand_kp)

    np.save(os.path.join(vid_folder, 'pose.npy'), pose_kp)
    np.save(os.path.join(vid_folder, 'face.npy'), face_kp)
    np.save(os.path.join(vid_folder, 'left_hand.npy'), left_hand_kp)
    np.save(os.path.join(vid_folder, 'right_hand.npy'), right_hand_kp)


vid_folder = "raw_videos"
key_points_path = "features/"
if not os.path.exists(key_points_path):
    os.makedirs(key_points_path)
vids = os.listdir(vid_folder)

vids_start = 2
vids_end = 2000


for vid in tqdm(vids[vids_start:vids_end]):
    vid_path = os.path.join(vid_folder, vid.replace('\\', '/'))
    try:
        feature_extractor_video(vid_path, key_points_path)
    except:
        print("Error with video: ", vid_path)
        continue