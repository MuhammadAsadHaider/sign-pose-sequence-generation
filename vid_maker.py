import cv2
import os

base_url = 'F:/Datasets/kp/_0fO5ETSwyg_3-5-rgb_front'
visualize_path = os.path.join(base_url, 'visualize')

# read in the frames and make the video
frames = []
for i in range(len(os.listdir(visualize_path))):
    frame = cv2.imread(os.path.join(visualize_path, f'{i}.png'))
    frames.append(frame)

height, width, layers = frames[0].shape
size = (width, height)

out = cv2.VideoWriter(os.path.join(visualize_path, 'video.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(frames)):
    out.write(frames[i])
out.release()
