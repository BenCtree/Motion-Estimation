# Motion-Estimation

An implementation in python of a block matching algorithm for motion estimation in a video clip.

First, a video clip is read in as a sequence of still frames using the cv2 library.

The algorithm splits a video frame F_i and the next frame F_i+1 into blocks. Then for each block in F_i, it finds the closest matching block in F_i+1 selected by minimising the root sum of squared distances between the origin block and each candidate block. Then, we draw a vector on F_i with the base starting at the origin block's centre and the tip pointing in the direction of the matched block's centre, with thickness based on how far away the matching block is.

We do this for each block in each frame and then output the resulting sequence of frames with vectors drawn on them as a video file.

We can choose a radius within which to perform this search - at the moment the radius is set to one block. We also choose a max and min threshold for the RSSD used to find a match - this filters out blocks for which nothing or very little has changed from frame to frame (eg the background of the image) or blocks for which the RSSD is very large (eg if the clip cuts from one shot to another). This helps filter out noise, making our resulting motion vectors drawn on the image better overlay only the moving parts of the image. Some experimentation is required to find the optimal t_max and t_min parameters to minimise noise.

The vector_monkey.mov file demonstrates the output of the algorithm. 

Let the motion estimation begin!
