import argparse
import os 
from tqdm import tqdm
from glob import glob 
import sys

from py_script.utils.utils import *

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg inference code'
    )

    parser.add_argument('gtFine_dir', help='folder path to gtFine')
    parser.add_argument('save_dir', help='save directory path')
    parser.add_argument('erosion_dilation', help=('label image transformation, choose one ["er", "di"]'))
    
    

    args = parser.parse_args()

    return args

def get_damage_palette (damage) :
    print(damage)
    if damage == "crack":
        damage = [[0, 0, 0], [255, 0, 0]]
        damage_idx = 1
    if damage == "efflorescence":
        damage = [[0, 0, 0], [0, 255, 0]]
        damage_idx = 2
    if damage == "rebar":
        damage = [[0, 0, 0], [255, 255, 0]]
        damage_idx = 3
    if damage == "spalling":
        damage = [[0, 0, 0], [0, 0, 255]]
        damage_idx = 4
    return damage, damage_idx


def main():
    args = parse_args()
    
    gtFine_dir = args.gtFine_dir

    label_list = glob(os.path.join(gtFine_dir, '*.png'))
    leftImg8bit_dir = gtFine_dir.replace("gtFine", "leftImg8bit")
    img_list = glob(os.path.join(leftImg8bit_dir, '*.png'))
    


    palette = np.array([[0, 0, 0],
               [0, 255, 0],
               [0, 255, 0],
               [0, 255, 255], 
               [255, 0, 0]])

    # set foler path to save image cityscape format
    color_save_path = os.path.join(args.save_dir, "color")
    gtFine_save_path = os.path.join(args.save_dir, "gtFine")
    leftImg8bit_save_path = os.path.join(args.save_dir, "leftImg8bit")
    os.makedirs(color_save_path, exist_ok=True)
    os.makedirs(gtFine_save_path, exist_ok=True)
    os.makedirs(leftImg8bit_save_path, exist_ok=True)
    

    for label, img in zip(tqdm(label_list), img_list):
        
        label_src = imread(label)
        img_src = imread(img)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)

        label_name = os.path.basename(label)
        img_name = os.path.basename(img)

        imwrite(os.path.join(leftImg8bit_save_path, img_name), img_src)
        
        
        label_transform = args.erosion_dilation
        if label_transform == "er":
            kernel = np.ones((3, 3), np.uint8)
            label_dst = cv2.erode(label_src, kernel, iterations=1)
        elif label_transform == "di":
            kernel = np.ones((3, 3), np.uint8)
            label_dst = cv2.dilate(label_src, kernel, iterations=1)
        
        imwrite(os.path.join(gtFine_save_path, label_name), label_dst)

        color_name = img_name.replace("_leftImg8bit.png", "_color.png")
        color_path = os.path.join(color_save_path, color_name)

        colormap = blendImageWithColorMap(image=img_src, label=label_dst, palette=palette, alpha=float(0.5))

        imwrite(color_path, colormap)
    




if __name__ == "__main__" :
    main()