import math

# Colors.
WHEAT1  = (186,231,255)    
SPRINGGREEN = (127,255,0)
ALICEBLUE  = (255,248,240)
AQUA  =  (255,255,0)
BLUE   = (255, 0, 0)
RED    = (0, 0, 255)
GREEN  = (0, 255, 0) 
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)

########################################################################################################
def is_intersecting(rectangle1, rectangle2):
	if (rectangle1[2] < rectangle2[0] or rectangle2[2] < rectangle1[0] or
		rectangle1[3] < rectangle2[1] or rectangle2[3] < rectangle1[1]):
		return False
	return True

'''def intersects(rectangle1, rectangle2):
    return not (rectangle1[0] + rectangle1[2] < other.bottom_left.x or self.bottom_left.x > other.top_right.x 
    	or self.top_right.y < other.bottom_left.y or self.bottom_left.y > other.top_right.y)'''
def is_close(rect1, rect2, distance_threshold):
	# Calculate the centers of the rectangles
	center1 = ((rect1[0] + rect1[2]) / 2, (rect1[1] + rect1[3]) / 2)
	center2 = ((rect2[0] + rect2[2]) / 2, (rect2[1] + rect2[3]) / 2)
	
	# Calculate the Euclidean distance between the centers
	distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

	
	
	# Return True if the distance is less than or equal to the threshold, False otherwise
	return distance <= distance_threshold

def is_close_camVscross(rect_cross, rect_cam, distance_threshold):
	# Calculate the centers of the rectangles
	i = False
	center1 = ((rect_cross[0] + rect_cross[2]) // 2, (rect_cross[3]) // 1)
	center2 = ((rect_cam[0] + rect_cam[2]) // 2, (rect_cam[1] + rect_cam[3]) // 2)
	
	# Calculate the Euclidean distance between the centers
	distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

	
	if(distance <= distance_threshold and (rect_cross[3] < rect_cam[1] or rect_cross[3] == rect_cam[3])):
		i =True
	# Return True if the distance is less than or equal to the threshold, False otherwise
	return i

#########################################################################################################

'''def is_cross_tl(class_name):
	rectangle1 = [P1_Cam[0], P1_Cam[1] ,P2_Cam[0], P2_Cam[1]]
	rectangle2 = [p1[0], p1[1] ,p2[0], p2[1]]
	IN_Cross = is_intersecting(rectangle1,rectangle2)  # CHECK IF camera IS IN cross line OF CAMERA
	if found_crosswalk and IN_Cross:
		if(found_red_light):
			rectangle1 = [P1_ROI[0], P1_ROI[1] ,P2_ROI[0], P2_ROI[1]]
			rectangle2 = [p1[0], p1[1] ,p2[0], p2[1]]
			INT = is_intersecting(rectangle1,rectangle2) # CHECK IF TRAFFIC LIGHT IS IN FRONT OF CAMERA
			direction = "Stop! it's red"
		else: # if there is the green light or if there is no traffic light
			#direction = "Go! it's green" 
			direction = Navigate(frame)


		# Give navigation orders based on direction and position relative to crosswalk
	else:
		direction = Navigate(frame)

	

	return '''



#########################################################################################################

def is_car_colsing(P1_Cam, P2_Cam, box_car, thr):
	i =False
	# Define the camera man pose 
	Cam_x1, Cam_y1, Cam_x2, Cam_y2 = P1_Cam[0], P1_Cam[1] ,P2_Cam[0], P2_Cam[1]

	# Calculate the center point of the Cam
	Cam_center_x = (Cam_x1 + Cam_x2) // 2
	Cam_center_y = (Cam_y1 + Cam_y2) // 2

	x, y, w, h = box_car
	x1,x2, y1, y2 = x, x+w, y, y+h

	# Calculate the center point of the car bounding box
	car_center_x = (x1 + x2) // 2
	car_center_y = (y1 + y2) // 2

	# Calculate the distance between the car center and Cam center
	distance = math.sqrt((car_center_x - Cam_center_x)**2 + (car_center_y - Cam_center_y)**2)
	#print(distance)
	if distance < thr:
		i = True
	return i

	'''# Check if the car bounding box intersects with the Cam
				if x1 < Cam_x2 and x2 > Cam_x1 and y1 < Cam_y2 and y2 > Cam_y1:
			'''
		
	

def ROI(W,H):
	P1_ROI = (W//4, 0)
	P2_ROI =(3*W//4, 3*H//4)
	return P1_ROI, P2_ROI

def Cam_man_pose(W,H):
	P1_Cam = (W//2 - 30, H-30)
	P2_Cam = (W//2 + 30, H)
	return P1_Cam, P2_Cam