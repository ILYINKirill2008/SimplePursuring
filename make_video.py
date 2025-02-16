import glob
import cv2
import os

img_array = []
    
print(os.getcwd())
print(glob.glob('frames/*.png'))
frame_files = sorted(glob.glob("frames/episode_*.png")) 


frame = cv2.imread(frame_files[0])
height, width, layers = frame.shape
size = (width, height)


out = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, size)

n = len(frame_files)
for i, filename in zip(range(n),frame_files):
    if i % 4 != 0:
        continue
    
    img = cv2.imread(filename)
    out.write(img)
    
out.release()
