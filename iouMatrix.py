"""
신속 라벨링 시스템의 성능 검증을 위한 비교실험
일관성 검증을 위한 라벨러 간의 gtFine 결과 비교 
"""


import os
import argparse

from glob import glob 
from tqdm import tqdm
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

from py_script.utils.utils import *

parser = argparse.ArgumentParser()

parser.add_argument("gtFine_1_path", help="folder path to gtFineImages", type=str)
parser.add_argument("gtFine_2_path", help="folder path to gtFineImages", type=str)
parser.add_argument("gtFine_3_path", help="folder path to gtFineImages", type=str)
parser.add_argument("gtFine_4_path", help="folder path to gtFineImages", type=str)

parser.add_argument("--target_class_num", default=1, type=int)

args = parser.parse_args()


def main():

    gtFine_1_path = args.gtFine_1_path
    gtFine_2_path = args.gtFine_2_path
    gtFine_3_path = args.gtFine_3_path
    gtFine_4_path = args.gtFine_4_path
    
    target_class_num = args.target_class_num

    if target_class_num == 1:
        target = "crack"
    elif target_class_num == 2:
        target = "efflorescence"
    elif target_class_num == 3:
        target = "rebar-exposure"
    elif target_class_num == 4:
        target = "spalling"


    gt_1_list = glob(os.path.join(gtFine_1_path, f'{target}', '*.png'))
    gt_2_list = glob(os.path.join(gtFine_2_path, f'{target}', '*.png'))
    gt_3_list = glob(os.path.join(gtFine_3_path, f'{target}', '*.png'))
    gt_4_list = glob(os.path.join(gtFine_4_path, f'{target}', '*.png'))
    
    IoU_list_1and2 = []
    IoU_list_1and3 = []
    IoU_list_1and4 = []
    IoU_list_2and3 = []
    IoU_list_2and4 = []
    IoU_list_3and4 = []
    
    
    for gt_1, gt_2, gt_3, gt_4 in zip(gt_1_list, gt_2_list, gt_3_list, gt_4_list):
        
        gt_1_img = imread(gt_1)
        gt_2_img = imread(gt_2)
        gt_3_img = imread(gt_3)
        gt_4_img = imread(gt_4)

        gt_1_img[gt_1_img != target_class_num] = 0
        gt_1_img[gt_1_img == target_class_num] = 1 

        gt_2_img[gt_2_img != target_class_num] = 0
        gt_2_img[gt_2_img == target_class_num] = 1 

        gt_3_img[gt_3_img != target_class_num] = 0
        gt_3_img[gt_3_img == target_class_num] = 1 

        gt_4_img[gt_4_img != target_class_num] = 0
        gt_4_img[gt_4_img == target_class_num] = 1 

        
        if [1] in np.unique(gt_1_img) and [1] in np.unique(gt_2_img) and [1] in np.unique(gt_3_img) and [1] in np.unique(gt_4_img):

            # calculate iou
            iou_1and2 = iou(gt_1_img, gt_2_img)
            iou_1and3 = iou(gt_1_img, gt_3_img)
            iou_1and4 = iou(gt_1_img, gt_4_img)
            iou_2and3 = iou(gt_2_img, gt_3_img)
            iou_2and4 = iou(gt_2_img, gt_4_img)
            iou_3and4 = iou(gt_3_img, gt_4_img)
            
            IoU_list_1and2.append(iou_1and2)
            IoU_list_1and3.append(iou_1and3) 
            IoU_list_1and4.append(iou_1and4)
            IoU_list_2and3.append(iou_2and3)
            IoU_list_2and4.append(iou_2and4)
            IoU_list_3and4.append(iou_3and4)           

            
            


    print(f"mIoU_1and2: {np.mean(IoU_list_1and2)} ({np.mean(IoU_list_1and2)*100}%)")
    print(f"mIoU_1and3: {np.mean(IoU_list_1and3)} ({np.mean(IoU_list_1and3)*100}%)")
    print(f"mIoU_1and4: {np.mean(IoU_list_1and4)} ({np.mean(IoU_list_1and4)*100}%)")
    print(f"mIoU_2and3: {np.mean(IoU_list_2and3)} ({np.mean(IoU_list_2and3)*100}%)")
    print(f"mIoU_2and4: {np.mean(IoU_list_2and4)} ({np.mean(IoU_list_2and4)*100}%)")
    print(f"mIoU_3and4: {np.mean(IoU_list_3and4)} ({np.mean(IoU_list_3and4)*100}%)")
    
    Total = [np.mean(IoU_list_1and2), np.mean(IoU_list_1and3), np.mean(IoU_list_1and4), np.mean(IoU_list_2and3), np.mean(IoU_list_2and4), np.mean(IoU_list_3and4)]
    
    print(f"Total mIoU: {np.mean(Total)} ({np.mean(Total)*100}%)")
            
            
            
            

            

def iou (gt_1_img, gt_2_img):
    
    # 교집합
    intersection = cv2.countNonZero(cv2.bitwise_and(gt_1_img, gt_2_img))
    # 합집합
    union = cv2.countNonZero(cv2.bitwise_or(gt_1_img, gt_2_img))
    
    IoU = intersection/union
    
    return IoU 




        
        

if __name__ == '__main__':
    

    main()
