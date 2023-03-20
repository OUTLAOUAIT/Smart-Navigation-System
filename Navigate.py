import cv2
import numpy as np

def Navigate(frame):
    StepSize = 5

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply a median blur to reduce noise
    gray = cv2.medianBlur(gray, 5)
    # Apply the Canny edge detection algorithm with lower and upper threshold values
    edges_ = cv2.Canny(gray, 50, 150)
    # Find contours of the edges
    contours, _ = cv2.findContours(edges_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by length
    min_length = 200
    filtered_contours = [c for c in contours if cv2.arcLength(c, True) > min_length]
    # Create a new binary image with just the filtered edges
    filtered_edges = np.zeros_like(edges_)
    cv2.drawContours(filtered_edges, filtered_contours, -1, 255, thickness=1)
    edges = filtered_edges
    ###################################

    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    edges = thresh+edges

    ###################################
    cv2.imshow('edges',edges)
    H, W = edges.shape
    img_h = H - 1
    img_w = W - 1

    EdgeArray = []
    for j in range(0, img_w, StepSize):
        pixel = (j,0)
        for i in range(img_h-5, 0, -1): ############################################## TRY OTHER THAN 5
            if edges.item(i, j) == 255:
                pixel = (j,i)
                break
        EdgeArray.append(pixel)

    chunks = np.array_split(EdgeArray, 3)

    #cv2.line(frame, (np.amax(chunks[0][:, 0]),0), (np.amax(chunks[0][:, 0]),H), BLUE, 2)
    #cv2.line(frame, (np.amax(chunks[1][:, 0]),0), (np.amax(chunks[1][:, 0]),H), BLACK, 2)
    #cv2.line(frame, (np.amax(chunks[2][:, 0]),0), (np.amax(chunks[2][:, 0]),H), RED, 2)
   
    midpoint = (W//2, H-30) 
    c = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # a list of RGB colors to use for each line
    

    for chunk, color in zip(chunks, colors):
        x_vals, y_vals = zip(*chunk)
        avg_x = int(np.mean(x_vals))
        avg_y = int(np.mean(y_vals))
        c.append([avg_y, avg_x])

        #cv2.line(frame, midpoint, (avg_x, avg_y), color, 2)

    #cv2.circle(frame, (c[0][1],c[0][0]), radius=3, color= BLUE, thickness=-1)
    #cv2.circle(frame, (c[1][1],c[1][0]), radius=3, color= BLACK, thickness=-1)
    #cv2.circle(frame, (c[2][1],c[2][0]), radius=3, color= RED, thickness=-1)

    forwardEdge = c[1]
    farthest_point = (min(c))

    dis = [c[0][0], c[1][0], c[2][0]]
    #cv2.line(frame,midpoint, (farthest_point[1],farthest_point[0]), (204,50, 153), 2)

    if np.argmin(dis) == 1:
        direction = "forward"
    elif np.argmin(dis) == 0:
        direction = "left"
    else:
        direction = "right"

    return direction
