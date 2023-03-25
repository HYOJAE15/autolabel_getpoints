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

parser.add_argument("groundTruth_path", help="folder path to gtFineImages for ground truth", type=str)
parser.add_argument("autoLabel_path", help="folder path to gtFineImages for auto label", type=str)
parser.add_argument("--target_class_num", default=1, type=int)

args = parser.parse_args()


def main():

    groundTruth_path = args.groundTruth_path
    autoLabel_path = args.autoLabel_path
    target_class_num = args.target_class_num

    grd_gtf_list = glob(os.path.join(groundTruth_path, '*', '*.png'))
    atl_gtf_list = glob(os.path.join(autoLabel_path, '*', '*.png'))

    IoU_list = []
    IoU_value_list = []
    show_only_iou_list = []

    for grd_path, atl_path in zip(grd_gtf_list, atl_gtf_list):
        # print(f"grd_path: {grd_path} atl_path: {atl_path}")

        # \ 역슬래시로 불러올때는 디코딩(cv2.imdecoding) 해줘야한다
        # / 정슬래시로 불러오면 그냥(cv.imread) 불러와진다 
        # 무슨 차이인가 이미지 읽기 참 어렵네
        
        grd_gtf = imread(grd_path)
        atl_gtf = imread(atl_path)

        grd_gtf[grd_gtf != target_class_num] = 0
        grd_gtf[grd_gtf == target_class_num] = 1 

        atl_gtf[atl_gtf != target_class_num] = 0
        atl_gtf[atl_gtf == target_class_num] = 1 

        
        if [1] in np.unique(grd_gtf) :

            # 교집합
            intersection = cv2.countNonZero(cv2.bitwise_and(grd_gtf, atl_gtf))
            
            # 합집합
            union = cv2.countNonZero(cv2.bitwise_or(grd_gtf, atl_gtf))

            IoU = intersection/union
            print(f"file: {os.path.basename(grd_path)},IoU: {IoU}")
            IoU_list.append([os.path.basename(grd_path), IoU])
            IoU_value_list.append(IoU)

            show_only_iou_list.append(IoU)
    
    # print(f"iou list: {IoU_list}")
    
    for i in list(range(len(show_only_iou_list))):
        print(show_only_iou_list[i])


    print(f"mIoU: {np.mean(IoU_value_list)}({np.mean(IoU_value_list)*100}%)")
            
            
            
            # 계산된 결과를 .csv 파일로 저장해줍니다.
            

            









        
        

if __name__ == '__main__':
    

    main()
