
import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models_
import torch
import torch.nn.functional as F
from PIL import Image
#from google.colab.patches import cv2_imshow
import torch
import torchvision.transforms as transforms


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]


def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_pretrained(model, pretrained): 
    pretrained_dict = torch.load(pretrained, map_location=device)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    '''msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')'''
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

model = models_.pidnet.get_pred_model('pidnet-l', 19 if True else 11)
model = load_pretrained(model, '/content/drive/MyDrive/PID/PIDNet_L_Cityscapes_test.pt').to(device)
model.eval()

def PID_Seg(image):

    StepSize = 5
    # Define the target size for resizing
    with torch.no_grad():
        img = image
        sv_img = np.zeros_like(img).astype(np.uint8)
        img = input_transform(img)
        img = img.transpose((2, 0, 1)).copy()
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred = model(img)
        pred = F.interpolate(pred, size=img.size()[-2:], 
                             mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        #print(pred)
        for i, color in enumerate(color_map):
            for j in range(3):
                sv_img[:,:,j][pred==i] = color_map[i][j]
        sv_img = Image.fromarray(sv_img)

        colorized_rgb = sv_img.convert('RGB')
        
        # Convert to BGR
        colorized_bgr = np.array(colorized_rgb)[:, :, ::-1].copy() # Rearrange color channels

        seg = colorized_bgr.copy()
        # Find the region with label == 1 (black)
        black_mask =(pred == 0) | (pred == 1)
        # Invert the mask so that black regions are white and other regions are black
        white_mask = ~black_mask
        
        # Draw the white regions as black and black regions as white
        colorized_bgr[white_mask] = (255, 255, 255)
        colorized_bgr[black_mask] = (0, 0, 0)

        gray = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2GRAY)
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

        #cv2_imshow(edges)

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

        for x in range(len(EdgeArray)-1):
              cv2.line(image, EdgeArray[x], EdgeArray[x+1], (0,255,0), 1)
        for x in range(len(EdgeArray)):
              cv2.line(image, (x*StepSize, img_h), EdgeArray[x],(0,255,0),1)

        cv2.line(image, (np.amax(chunks[0][:, 0]),0), (np.amax(chunks[0][:, 0]),H), (255, 0, 0), 2)
        cv2.line(image, (np.amax(chunks[1][:, 0]),0), (np.amax(chunks[1][:, 0]),H), (0, 0, 255), 2)
        #cv2.line(image, (np.amax(chunks[2][:, 0]),0), (np.amax(chunks[2][:, 0]),H), (0, 0, 255), 2)
        
        midpoint = (W//2, H-5) 
        c = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # a list of RGB colors to use for each line
        

        for chunk, color in zip(chunks, colors):
            x_vals, y_vals = zip(*chunk)
            avg_x = int(np.mean(x_vals))
            avg_y = int(np.mean(y_vals))
            c.append([avg_y, avg_x])

            #cv2.line(frame, midpoint, (avg_x, avg_y), color, 2)

        cv2.circle(image, (c[0][1],c[0][0]), radius=3, color= colors[0], thickness=-1)
        cv2.circle(image, (c[1][1],c[1][0]), radius=3, color= colors[1], thickness=-1)
        cv2.circle(image, (c[2][1],c[2][0]), radius=3, color= colors[2], thickness=-1)

        forwardEdge = c[1]
        farthest_point = (min(c))

        dis = [c[0][0], c[1][0], c[2][0]]
        #print(dis)
        cv2.line(image,midpoint, (farthest_point[1],farthest_point[0]), (204,50, 153), 2)

        if np.argmin(dis) == 1:
            direction = "forward"
        elif np.argmin(dis) == 0:
            direction = "left"
        else:
            direction = "right"

        return image, direction, seg


