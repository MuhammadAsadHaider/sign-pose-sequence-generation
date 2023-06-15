import os
import cv2

video_path = "F:/Datasets/raw_videos/_20g7MG8K1U_24-8-rgb_front.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

print('fps: ', fps)

cap.release()

cv2.destroyAllWindows()

