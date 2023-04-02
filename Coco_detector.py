import cv2
import numpy as np
import math
import time

def Coco_detector(Model_path_weights, Model_path_cfg, class_idxs, frame):
    H, W = frame.shape[:2]
    
    net = cv2.dnn.readNet(Model_path_weights,Model_path_cfg)
    # Prepare the input to the model by converting it to a blob
    # AS SIZE WE CAN TRY ALSO (320,320)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (160, 160), swapRB=True, crop=False)
    # Run the forward pass of the model
    net.setInput(blob)
    layer_names = net.getLayerNames()
    outputs = net.forward([layer_names[i - 1] for i in net.getUnconnectedOutLayers()])
    # Post-process the output to get the bounding boxes, confidences, and class predictions
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id in class_idxs:
                confidence = scores[class_id]
                if confidence > 0.6:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    # Perform non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    return indices, boxes, class_ids






