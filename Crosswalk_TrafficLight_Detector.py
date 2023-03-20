import torch
import numpy as np
import cv2
from my_utils.utils import letterbox, non_max_suppression, scale_coords

def CrosswalkTrafficLight(model, img, device, stride):

    #img = cv2.resize(img, (1000,600))
    img_input = letterbox(img, 640, stride=stride)[0]
    img_input = img_input.transpose((2, 0, 1))[::-1]
    img_input = np.ascontiguousarray(img_input)
    img_input = torch.from_numpy(img_input).to(device)
    img_input = img_input.float()
    img_input /= 255.
    img_input = img_input.unsqueeze(0)

    # inference
    pred = model(img_input, augment=False, visualize=False)[0]

    '''# postprocess
                conf_thres = 0.5  # confidence threshold
                iou_thres = 0.45  # NMS IOU threshold
                max_det = 1000  # maximum detections per image
                classes = None  # filter by class
                agnostic_nms = False  # class-agnostic NMS'''

    pred = non_max_suppression(pred, 0.5, 0.45, None, False, max_det=1000)[0]
    pred = pred.cpu().numpy()
    pred[:, :4] = scale_coords(img_input.shape[2:], pred[:, :4], img.shape).round()

    return pred, img













