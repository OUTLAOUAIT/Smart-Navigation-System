import cv2
import numpy as np
import math
import time
import torch
from Coco_detector import Coco_detector
from Crosswalk_TrafficLight_Detector import CrosswalkTrafficLight
from Navigate import Navigate
from Config import WHEAT1, SPRINGGREEN, RED, GREEN,BLACK, BLUE, YELLOW, is_intersecting, is_close, is_car_colsing, is_close_camVscross,ROI, Cam_man_pose
from PID_segmentation import PID_Seg

start = time.time()
video = "/content/drive/MyDrive/Videos/naples.mp4"

Model_path_weights = "Model/yolov7-tiny.weights"
Model_path_cfg = "Model/yolov7-tiny.cfg"
model = 'Model/best.pt'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(model, map_location=device)
model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
class_names = ['crosswalk', 'red light', 'green light']  # model.names
stride = int(model.stride.max())
colors = ((50, 50, 50),(255,0 , 0),(0, 0, 255)) # (gray, red, green)


# Set the subsampling rate to 10
subsampling_rate = 10



# Load the class names
classes = []
with open("Model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

'''for i in ['person', 'bicycle', 'car', 'motorbike', 'bus', 'train', 'truck']:
    print(classes.index(i))'''

class_idxs = [0,1,2,3,5,6,7]

# Load the video file
cap = cv2.VideoCapture(video)
StepSize = 5

ret, frame = cap.read()
if ret:
    H, W = frame.shape[:2]  # Get the height and width of the frame

out = cv2.VideoWriter('/content/drive/MyDrive/Results/output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (W, H))
# Process every 10th frame of the input video
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        frame_copy = frame.copy()
        break

    # Only process every 10th frame
    if frame_count % subsampling_rate == 0:

        pred, frame = CrosswalkTrafficLight(model,frame, device, stride)
        indices, boxes, class_ids  = Coco_detector(Model_path_weights, Model_path_cfg, class_idxs, frame)
        H, W = frame.shape[:2]

        P1_ROI, P2_ROI =ROI(W,H)
        P1_Cam, P2_Cam = Cam_man_pose(W,H)

        #cv2.rectangle(frame, P1_ROI, P2_ROI, BLACK, 2)
        #cv2.rectangle(frame, P1_Cam, P2_Cam, RED, 2)

        rectangle_camera_man = [P1_Cam[0], P1_Cam[1] ,P2_Cam[0], P2_Cam[1]]
        rectangle_roi = [P1_ROI[0], P1_ROI[1] ,P2_ROI[0], P2_ROI[1]]

        # Draw bounding boxes around the detected objects
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            
            label = f"{classes[class_ids[i]]}" # Without PRINTING THE CONFIDENCES
            #print(class_ids[i])
            if(label in ['bicycle', 'car', 'motorbike', 'bus', 'train', 'truck']):
                # Check if the car bounding box intersects with the Camera man
                if is_car_colsing(P1_Cam, P2_Cam, box, 300):
                    message = 'A car is very close to you'
                    cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, SPRINGGREEN, 2, cv2.LINE_AA)
 
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2)
        
        # Initialize some variables
        crossing_lines_detected = False
        traffic_light_detected = False
        traffic_light_color = None

        for p in pred:
            class_name = class_names[int(p[5])]

            x1, y1, x2, y2 = [int(i) for i in p[:4]]
            p1, p2 = (x1, y1), (x2, y2)

            confidence = p[4]

            if class_name == 'crosswalk' and confidence > 0.5:
                # Camera man is close to the crossing lines
                crossing_lines_detected = True
                x1, y1, x2, y2 = [int(i) for i in p[:4]]
                pc1, pc2 = (x1, y1), (x2, y2)
                rectangle_cross = [pc1[0], pc1[1] ,pc2[0], pc2[1]]
                #cv2.rectangle(frame, pc1,pc2, (255,255,255), 2)

            elif class_name in ['red light', 'green light'] and confidence > 0.5:
                # Traffic light detected
                traffic_light_detected = True
                traffic_light_color = class_name

        if crossing_lines_detected:
            #print(is_intersecting(rectangle_cross, rectangle_camera_man))
            if is_close_camVscross(rectangle_cross,rectangle_camera_man,200):
                #cv2.putText(frame, "Close to Cross", (W//2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
                if(traffic_light_detected):
                    if traffic_light_color == 'red light':
                        order = 'stop'
                    else:
                        order = 'go ' + PID_Seg(frame)[1]# Navigate(frame)

            elif is_intersecting(rectangle_cross, rectangle_camera_man):
                #print(rectangle_cross)
                #print(rectangle_camera_man)
                #cv2.putText(frame, "On the cross", (W//2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
                order = 'go forward'# + Navigate(frame)
            else :
                order = 'go ' + PID_Seg(frame)[1]# Navigate(frame)
        else:
            order = PID_Seg(frame)[1]# Navigate(frame)

        cv2.putText(frame, order, (W//2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

        cv2.imshow("frame", frame) 
        out.write(frame) # write the frame to the output video

    frame_count += 1


    k = cv2.waitKey(5) & 0xFF  ##change to 5

    if k == 27:
       break

cap.release()
out.release() # release the VideoWriter object
cv2.destroyAllWindows()


end = time.time()
# print the difference between start and end time in milli. secs
print("The time of execution of above program is :",(end-start) * 10**3, "ms")


