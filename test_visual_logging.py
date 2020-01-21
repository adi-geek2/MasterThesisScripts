#import the necessary packages
from logging import FileHandler
from vlogging import VisualRecord
import logging
import cv2
import glob
import numpy as np
 
# open the logging file
logger = logging.getLogger("visual_logging_example")
fh = FileHandler("demo.html", mode = "w")
 
# set the logger attributes
logger.setLevel(logging.DEBUG)
logger.addHandler(fh)
 
# load our example image and convert it to grayscale
for img_key in glob.glob("./*.png"):
    image = cv2.imread(img_key)
    image_to_be_processed = image  # Input image
    
    print("running logger")
    image_to_be_processed = cv2.cvtColor(image_to_be_processed, cv2.COLOR_RGBA2RGB)
    if image_to_be_processed == None: 
            raise Exception("could not load input image !")
    # Color based segmentation for yellow color
    lower_boundary_yellow_color = (0, 200, 200)
    upper_boundary_yellow_color = (0, 240, 240)
    yellow_filter_mask = cv2.inRange(image_to_be_processed, lower_boundary_yellow_color, upper_boundary_yellow_color)
    # Refining and smoothing the boundaries of yellow color filter mask using morphology operations
    kernel = np.ones((3, 3), dtype=np.float32)
    yellow_filter_mask = cv2.morphologyEx(yellow_filter_mask, cv2.MORPH_OPEN, kernel)
    yellow_filter_mask = cv2.dilate(yellow_filter_mask, kernel, iterations=1)
    # Applying the mask to input image
    masked_image = cv2.bitwise_and(image_to_be_processed, image_to_be_processed, mask=yellow_filter_mask)
    # Module : Extraction of traffic sign bounding boxes for each segmented image
    # Convert to gray
    gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('gray' + str(img_key) + '.png', gray_masked_image)
    # Find edges
    edge_segmented_image = cv2.Canny(gray_masked_image, 150, 170)
    #cv2.imwrite('edge' + str(img_key) + '.png', edge_segmented_image)
    # Find contours : using RETR_EXTERNAL to fetch only external boundary
    _, contours, _ = cv2.findContours(edge_segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    # print(type(boundRect))
    # centers = [None] * len(contours)
    # radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        # Add the timestamp to bounding box parameter list
        # print(type(boundRect[i]))
        # list(boundRect[i])
        # boundRect[i].append(timestamp)
        #bb_param_lst_img.append(boundRect[i])
    # Add the new bounding box to list of existing bounding boxes for the image
    logger.debug(VisualRecord("Input Image = %s" % (img_key),
        [image_to_be_processed, masked_image, gray_masked_image, edge_segmented_image], fmt = "png"))