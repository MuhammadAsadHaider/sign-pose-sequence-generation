import mediapipe as mp
import cv2
import numpy as np
import os
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

sample = np.load('sample_0.npy', allow_pickle=True)
sample = np.reshape(sample, (250, 126))

total_frames = sample.shape[0]

right_hand = sample[:, :63].reshape(total_frames, 21, 3)
left_hand = sample[:, 63:].reshape(total_frames, 21, 3)

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

    # for kp in pose[i]:
    #     p.landmark.add(x=kp[0], y=kp[1], z=kp[2], visibility=kp[3])

    # for kp in face[i]:
    #     f.landmark.add(x=kp[0], y=kp[1], z=kp[2])

    for kp in left_hand[i]:
        lh.landmark.add(x=kp[0], y=kp[1], z=kp[2])

    for kp in right_hand[i]:
        rh.landmark.add(x=kp[0], y=kp[1], z=kp[2])
    
    white_background = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    draw_styled_landmarks(white_background, p, f, lh, rh)
    visualize_path = 'visualize'
    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)
    if not cv2.imwrite(os.path.join(visualize_path, f'{i}.png'), white_background):
        raise Exception("Could not write image")