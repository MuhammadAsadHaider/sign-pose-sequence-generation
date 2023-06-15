import mediapipe as mp
import cv2
import numpy as np
import os
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

base_url = 'F:/Datasets/features/EmxarfGOKd0_0-8-rgb_front'

pose = np.load(os.path.join(base_url, 'pose.npy'), allow_pickle=True)
face = np.load(os.path.join(base_url, 'face.npy'), allow_pickle=True)
left_hand = np.load(os.path.join(base_url, 'left_hand.npy'), allow_pickle=True)
right_hand = np.load(os.path.join(base_url, 'right_hand.npy'), allow_pickle=True)

total_frames = pose.shape[0]


def draw_styled_landmarks(image, pose, face, left_hand, right_hand):
    # Draw face connections
    # mp_drawing.draw_landmarks(image, face, mp_holistic.FACEMESH_TESSELATION, 
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          ) 
    # # Draw pose connections
    # mp_drawing.draw_landmarks(image, pose, mp_holistic.POSE_CONNECTIONS,
    #                          mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=1), 
    #                          mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1)
    #                          ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, left_hand, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=1)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, right_hand, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1)
                             ) 

for i in tqdm(range(total_frames)):
    p = mp.solutions.drawing_utils.landmark_pb2.NormalizedLandmarkList()
    f = mp.solutions.drawing_utils.landmark_pb2.NormalizedLandmarkList()
    lh = mp.solutions.drawing_utils.landmark_pb2.NormalizedLandmarkList()
    rh = mp.solutions.drawing_utils.landmark_pb2.NormalizedLandmarkList()

    for kp in pose[i]:
        p.landmark.add(x=kp[0], y=kp[1], z=kp[2], visibility=kp[3])

    for kp in face[i]:
        f.landmark.add(x=kp[0], y=kp[1], z=kp[2])

    for kp in left_hand[i]:
        lh.landmark.add(x=kp[0], y=kp[1], z=kp[2])

    for kp in right_hand[i]:
        rh.landmark.add(x=kp[0], y=kp[1], z=kp[2])
    
    white_background = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    draw_styled_landmarks(white_background, p, f, lh, rh)
    visualize_path = os.path.join(base_url, 'visualize')
    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)
    if not cv2.imwrite(os.path.join(base_url, 'visualize', f'{i}.png'), white_background):
        raise Exception("Could not write image")