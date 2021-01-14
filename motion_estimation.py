# Motion Estimation and Visualization

import sys
import os
import cv2
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from IPython.display import clear_output

# Capture Frames of Video

path_to_video = './monkey.avi'
frame_save_path = './frames/'
#path_to_video = './camels.mp4'
#frame_save_path = './frames2/'

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

cap = cv2.VideoCapture(path_to_video)
create_dir_if_not_exists(frame_save_path) # Or can create it manully
if not cap.isOpened():
    print('{} not opened'.format(path_to_video))
    sys.exit(1)
time_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_counter = 0                                             # FRAME_COUNTER
while(1):
    return_flag, frame = cap.read()  
    if not return_flag:
        print('Video Reach End')
        break
    # Main Content - Start
    cv2.imshow('VideoWindowTitle-Monkey', frame)
    cv2.imwrite(frame_save_path + 'frame%d.tif' % frame_counter, frame)
    frame_counter += 1
    # Main Content - End    
    if cv2.waitKey(30) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Check frame height, width and counter

print("frame height: ", frame_height)
print("frame width: ", frame_width)
print("frame counter: ", frame_counter)
print("time length: ", time_length)

# Set Block Size and Number of Blocks

k = 4
block_height = 2*k + 1
block_width = 2*k + 1
num_blocks = (frame_height*frame_width)/(block_height * block_width)

print(num_blocks)

# Helper Function: Root Sum of Squared Distances

def RSSD(source, candidate):
    diff = np.subtract(source, candidate)
    sqdiff = np.square(diff)
    return np.sqrt(sqdiff.sum())

# Block Matching Algorithm

end = frame_counter - 1
start = 0

# Search Radius for Target Block eg 3 = search within 3 block radius
radius = 1
# Displacement thresholds
t_min = 150
t_max = 160

while start <= end:
    
    # Read Current Frame
    current_frame = cv2.imread('frames/frame%d.tif' % start)
    # Read Next Frame
    plus1 = start + 1
    if plus1 <= frame_counter-1:
        next_frame = cv2.imread('frames/frame%d.tif' % plus1)
    else:
        next_frame = current_frame

    print('Processing frame: %d, overall progress: %.2f %%' % (start, start/end*100))
    clear_output(wait=True)
    
    # Counters for disp_vector indexes
    counter_x = 0
    counter_y = 0
    
    # Target Block Search (x = row index, y = column index)
    
    for x in range(0, frame_height, block_height):
        for y in range(0, frame_width, block_width):
            
            # Source Centroid
            sc_x = x + k
            sc_y = y + k
            
            source_block = current_frame[x:x+block_height, y:y+block_width, 0:3]
            
            # Using radius (only searches within certain num of blocks around x,y coords)
            if x-(block_height*radius) >= 0:
                start_xrad = x-(block_height*radius)
            else:
                start_xrad = 0
            if x+(block_height + block_height*radius) <= 576:
                end_xrad = x+(block_height + block_height*radius)
            else:
                end_xrad = 576
            if y-(block_width*radius) >= 0:
                start_yrad = y-(block_width*radius)
            else:
                start_yrad = 0
            if y+(block_width + block_width*radius) <= 576:
                end_yrad = y+(block_width + block_width*radius)
            else:
                end_yrad = 576
            
            # Dict with the RSSD between source and a candidate block as key
            # Coordinates for top left entry of that candidate block as value
            differences = {}
            
            # Search all candidate blocks in next frame (within radius) for target block
            # Target block has min RSSD out of all candidate blocks
            # (i = row index, j = column index)
            for i in range(start_xrad, end_xrad, block_height):
                for j in range(start_yrad, end_yrad, block_width):
                    
                    candidate_block = next_frame[i:i+block_height, j:j+block_width, 0:3]
                    
                    diff = RSSD(source_block, candidate_block)
                    
                    if diff not in differences:
                        differences[diff] = (i,j)
            
            #print(differences)
            
            # Find candidate with min RSSD from source - this is target block
            if len(differences) > 0:
                min_diff = min(differences)
            else:
                continue
            
            # Set thickness of vector, indicating intensity
            quarter = (t_max - t_min) / 4
            thickness = 0
            if t_min <= min_diff < (t_min + quarter):
                thickness = 2
            elif (t_min + quarter) <= min_diff < (t_min + quarter*2):
                thickness = 5
            elif (t_min + quarter*2) <= min_diff < (t_min + quarter*3):
                thickness = 8
            elif (t_min + quarter*3) <= min_diff <= t_max:
                thickness = 11
            
            # Bound by (t_min, t_max)
            if min_diff < t_max and min_diff > t_min:
                # target is tuple of coords for top left corner of target block
                target = differences[min_diff]
                
                # Target Centroid
                tc_x = target[0] + k
                tc_y = target[1] + k
                
                # Calculate displacement vector
                disp_x = tc_x - sc_x
                disp_y = tc_y - sc_y
                
                # Draw motion vector on current frame
                if disp_x != 0 and disp_y != 0:
                    # Swap x and y... x = row index (y axis), y = col index (x axis)
                    vector_frame = cv2.arrowedLine(current_frame, (sc_y, sc_x), (tc_y, tc_x), (200, 200, 200), thickness, tipLength = 1) 
            else:
                vector_frame = current_frame
            
            if counter_y < int(frame_width/block_width)-1:
                counter_y += 1

        counter_x += 1
        counter_y = 0
    
    create_dir_if_not_exists('./vectorframes/')
    #cv2.imshow('VideoWindowTitle-VectorFrame', vector_frame)
    cv2.imwrite('vectorframes/vf%d.tif' % start, vector_frame)
            
    if cv2.waitKey(30) & 0xff == ord('q'):
        break
    start += 1


# Save Rendered Frames as Video File

# Read the frames from 'vectorframe' folder and covert them into an output video

frame_load_path = './vectorframes/'
path_to_output_video = './vector_monkey.mov'
print(os.path.exists(frame_load_path))

out = cv2.VideoWriter(path_to_output_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (int(frame_width), int(frame_height)))

frame_counter = 0
while(1): #cap.isOpened():
    img = cv2.imread(frame_load_path + 'vf%d.tif' % frame_counter)
    if img is None:
        print('No more frames to be loaded')
        break;
    out.write(img)
    frame_counter += 1
out.release()
cv2.destroyAllWindows()

 