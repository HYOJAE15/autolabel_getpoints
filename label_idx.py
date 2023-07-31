import os
import csv

from py_script.utils.utils import *

import sys
import argparse
from tqdm import tqdm
from glob import glob 

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg inference code'
    )
    parser.add_argument('gtFine', help='gtFine Folder path')
    # parser.add_argument('src_idx', help='change label index')
    # parser.add_argument('dst_idx', help='changed label index')
    
    args = parser.parse_args()

    return args


def main():
    
    args = parse_args()

    gt_folder_path = args.gtFine
    # src_idx = args.src_idx
    # dst_idx = args.dst_idx

    # gt_list = glob(os.path.join(gt_folder_path, "*", "*.png"))
    gt_list = glob(os.path.join(gt_folder_path, "*.png"))

    for gt in tqdm(gt_list) :
        gt_img = imread(gt)

        # 신속 라벨링 시스템 ver.2.0 에서 라벨링 인덱스가 1과 255가 섞여서 나온다는 이슈 (23.06.02)
        gt_img[gt_img == 3] = 2
        # gt_img[gt_img == 1] = 1

        imwrite(gt, gt_img)


if __name__ == '__main__':
    main()