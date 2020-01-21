# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:06:39 2020

@author: Adi
"""
import csv
import shutil, os
#import numpy

def image_file_name_rerader(myfilepath):
    line_number = 0
    image_file_name_list_local = []     
    with open(myfilepath, 'rb') as f:
        mycsv = csv.reader(f,delimiter=';')
        mycsv = list(mycsv)
        for line_number in range(len(mycsv)):
            text = mycsv[line_number][0]
            image_file_name_list_local.append(text)
    print (line_number)
    return image_file_name_list_local
        
def copy_file_using_name_to_dest_folder(files):
    cwd= os.getcwd()
    print(cwd)
    os.makedirs('test_gt')
    for f in files:
        shutil.copy(f, 'test_gt')    
        
if __name__ == '__main__':
    #Read from csv file
    print(os.getcwd())
    rel_path_label = os.path.join('.', 'annotations')
    os.chdir(rel_path_label)
    myfilepath = 'test_labels.csv'
    image_file_name_list_main = image_file_name_rerader(myfilepath)
    
    #Copy image files containing objects
    rel_path_images = os.path.join('..', 'gt_images')
    os.chdir(rel_path_images)
    print(os.getcwd())
    copy_file_using_name_to_dest_folder(image_file_name_list_main) 