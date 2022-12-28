import os
import argparse

from glob import glob 
from tqdm import tqdm
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import *

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
            print(f"file: {os.path.basename(grd_path)}   IoU: {IoU}")
            
            # 계산된 결과를 .csv 파일로 저장해줍니다.
            

            









        
        # plt.imshow(grd_gtf)
        # plt.show()

        # atl_path = grd_path
        # atl_path = atl_path.replace()


# E:\Documents\1003_토목학회발표자료\experiment\autolabel_experiment_groundTruth\gtFine\crack\101_07cb38b4-3369-4aa7-b144-dfb55042f177_gtFine_labelIds.png

# # Ground Truth 이미지를 불러옵니다.
# grd_truth = cv2.imread('D:/IouCalculate/groundTruth/leftImg8bit/191210_0001_leftImg8bit.png', cv2.IMREAD_UNCHANGED)

# grd_truth_gray = cv2.cvtColor(grd_truth, cv2.COLOR_BGR2GRAY)

# plt.imshow(grd_truth_gray)
# plt.show()

# detected_img = cv2.imread('', cv2.IMREAD_UNCHANGED)


if __name__ == '__main__':
    

    main()
