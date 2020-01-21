#!/usr/bin/env python
# External includes
from PIL import Image
# from sensor_msgs.msg import CameraInfo, Image
# from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
# import rospy
import numpy as np
import random as rng
import time
import csv
import glob

# import argparse
# Global variables
start_time = time.time()
bb_param_array_final_list_of_list = []
# loop iterator
no_of_image_processed = 0
no_of_images_not_processed = 0

#bb_param_lst_img = []  # Initialize list of bounding boxes for a single image
bb_param_lst_all_images = []  # Initialize list of bounding boxes in all images as a list
# Data type definition for creating a mixed numpy array holding bounding box parameters
mtype = 'object'


# Random number generation used for lines of bounding box rectangles
# rng.seed(12345)


def calculate_bb(ground_truth_image, time_stamp):
    #print(ground_truth_image.shape)
    #print(os.getcwd())
    image_to_be_processed = ground_truth_image  # Input image
    #logger.debug(VisualRecord(("Input Image = %d" % (s)),
    #    [ground_truth_image], fmt = "png"))
    #print("running logger")
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
    #cv2.imwrite('gray' + str(time_stamp) + '.png', gray_masked_image)
    # Find edges
    edge_segmented_image = cv2.Canny(gray_masked_image, 150, 170)
    #cv2.imwrite('edge' + str(time_stamp) + '.png', edge_segmented_image)
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
    return boundRect


def save_bb_in_csv(bb_param_lst_img):
    os.chdir('/home/adeshpand/Dokumente/tensorflow/models/annotations')
    #print(os.getcwd())
    with open('train_labels.csv', mode='w') as label_file:
        label_writer = csv.writer(label_file)  # , delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        label_writer.writerow(["filename","width","height","class","xmin","ymin","xmax","ymax"])
        for row in bb_param_lst_img:
            label_writer.writerow(row)


if __name__ == '__main__':
    # parse arguments from OS
    # parser = argparse.ArgumentParser(description="Convert ground truth semantic segmentation images into csv")
    # parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing the images')
    # args = parser.parse_args()
    gt_dir = '../gt_images/train_gt/'
    print(os.getcwd())

    for img_key  in os.listdir(gt_dir):
        #Read input image
        print(str(gt_dir+ str(img_key)))
        image_gt = cv2.imread(gt_dir + img_key)
        if image_gt == None: 
            raise Exception("could not load image !")
        # get the address from OS to open the input image data
        # os.chdir('./gt')
        # Initializing array to hold the bb params
        bb_param_array_final = np.ones((1, 8), dtype=mtype)
        bb_param_array_final_list = []
        # print(bb_param_array_final.shape)
        # Iterator for placing timestamps
        # time_stamp_iterator = 0
        time_stamp = img_key  # Store the time stamp on the image from the image name
        bb_param_array_final[0][4] = time_stamp
        bb_param_array_final_list.append(bb_param_array_final[0][4])
        # Read and store dimensions of the image
        # height, width, channels = image_gt.shape
        height = 600
        width = 800
        # Variable to store object class . here it is constantly traffic_sign
        class_ = 'traffic_sign' 
        # print(bb_param_array_final.shape)
        # image_gt = cv2.imread(args.directory + img_key)
        # print('imagename = ', img_key)
        
        #image_gt = cv2.imread(img_key)
        # trap to prevent errors from images that are not containing traffic signs
        try:
            bb_param_tuple = calculate_bb(image_gt, time_stamp)
            #print('length = ', len(bb_param_tuple))
            #print('tuple returned from bbox calc module = ', bb_param_tuple)
            # print(bb_param_tuple[row_index])
            bb_param_tuple_coverted_to_array = np.array(bb_param_tuple)
            #print('bounding box array = ', bb_param_tuple_coverted_to_array)
            # print(bb_param_tuple_array)
            bb_param_array_final_list.append(width)
            bb_param_array_final_list.append(height)
            bb_param_array_final_list.append(class_)
            bb_param_array_final[0][0] = bb_param_tuple_coverted_to_array[0][0]
            bb_param_array_final_list.append(bb_param_array_final[0][0])
            bb_param_array_final[0][1] = bb_param_tuple_coverted_to_array[0][1]
            bb_param_array_final_list.append(bb_param_array_final[0][1])
            #Calculate co-ordinates of bottom and right terminal pixels of  bounding box
            bb_param_array_final[0][2] = bb_param_tuple_coverted_to_array[0][2]
            xmax = bb_param_array_final[0][0] + bb_param_array_final[0][2]
            bb_param_array_final_list.append(xmax)
            bb_param_array_final[0][3] = bb_param_tuple_coverted_to_array[0][3]
            ymax = bb_param_array_final[0][1] + bb_param_array_final[0][3]
            bb_param_array_final_list.append(ymax)
            no_of_image_processed = no_of_image_processed + 1
            # print(bb_param_array_final_list)
            # bb_param_array_final_list_of_list.append(bb_param_array_final_list)
            bb_param_array_final_list_of_list.append(bb_param_array_final_list)
            # bb_param_array_final_list.append()
            # bb_param_array_final_list_of_list =
            # print(bb_param_array_final)
            # bb_param_lst_all_images.append(bb_param_array_final)
        except:
            no_of_images_not_processed = no_of_images_not_processed + 1
            pass
    #print('final bounding box parameters of all images', bb_param_array_final_list_of_list)
    print('Number of images not processed = ', no_of_images_not_processed)
    print('Number of image processed =', no_of_image_processed)
    #print('Number of images present in directory', len(os.listdir(gt_dir)))
    save_bb_in_csv(bb_param_array_final_list_of_list)
    # Calculate total execution time
    print("--- %s seconds ---" % (time.time() - start_time))
