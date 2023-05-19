import os
import csv

from py_script.utils.utils import *

import sys
import argparse
from tqdm import tqdm
from glob import glob 

sys.path.append("./dnn/mmsegmentation")
from mmseg.apis import init_segmentor, inference_segmentor

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg inference code'
    )
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'img_dir',
        help=('image directory for inference'))
    parser.add_argument(
        'point_dir',
        help=('point csv file directory'))
    parser.add_argument(
        'save_dir',
        help=('directory to save result'))
    parser.add_argument(
        '--damage_type',
        default='crack',
        help=('damage type to be inference'))
    parser.add_argument(
        '--image_preprocessing',
        default='None',
        help=('image preprocessing histogram equalization [None, gr, hsv, ycc]'))
    

    args = parser.parse_args()

    return args

def get_damage_palette (damage) :
    if damage == "crack":
        damage = (0, 0, 255)
        damage_idx = 1
    if damage == "efflorescence":
        damage = (0, 255, 0)
        damage_idx = 2
    if damage == "rebar-exposure":
        damage = (0, 255, 255)
        damage_idx = 3
    if damage == "spalling":
        damage = (255, 0, 0)
        damage_idx = 4
    return damage, damage_idx

def main() :

    args = parse_args()

    palette = np.array([[0, 0, 0],
               [0, 0, 255],
               [0, 255, 0],
               [0, 255, 255], 
               [255, 0, 0]])

    img_pre = args.image_preprocessing
    
    img_list = glob(os.path.join(args.img_dir, f"{args.damage_type}", '*.png'))
    
    if args.damage_type == "rebar-exposure" :
        rebarExposure = "rebarExposure"
        point_list = glob(os.path.join(args.point_dir, f'{rebarExposure}*'))
    else :
        point_list = glob(os.path.join(args.point_dir, f'{args.damage_type}*'))

    # print(point_list)
    # for img in img_list :
    #     src = imread(img)
    
    model = init_segmentor(args.config, args.checkpoint, device='cuda:0')


    for point in tqdm(point_list) :

        getPointsList = []

        csv_basename = os.path.basename(point)
        img_8bit_basename = csv_basename.replace(".csv", "_leftImg8bit.png")
        img_path = os.path.join(args.img_dir, f"{args.damage_type}" ,img_8bit_basename)
        src = imread(img_path)
        src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        
        gt_path = make_cityscapes_format(img_path, args.save_dir)
        
        
        with open(point, "r", encoding="cp949", newline='') as f :
            data = csv.reader(f)
            for row in data:
                getPointsList.append(row)
        
        damage, damage_idx = get_damage_palette (args.damage_type)

        for idx in getPointsList:
            
            src_label_pointing = imread(gt_path)
            
            label = pointsRoi(model=model, 
                              src=src, 
                              label=src_label_pointing, 
                              label_segmentation=damage_idx, 
                              y_start=int(idx[0]), 
                              y_end=int(idx[1]), 
                              x_start=int(idx[2]), 
                              x_end=int(idx[3]), 
                              )
            
            imwrite(gt_path, label)
        
        color_dir_path = os.path.join(args.save_dir, "color")
        os.makedirs(color_dir_path, exist_ok=True)
        img_color_basename = csv_basename.replace(".csv", "_color.png")
        color_path = os.path.join(color_dir_path, img_color_basename)
        
        dst_label = imread(gt_path)
            
        for idx, point in enumerate(getPointsList, start=1):
            
            if idx == 1 :
                colormap = blendImageWithColorMap(image=src, label=dst_label, palette=palette, alpha=float(0.5))
                
                
                rect_start = (int(point[2]), int(point[0]))
                rect_end = (int(point[3]), int(point[1]))
                rect = cv2.rectangle(colormap, rect_start, rect_end, damage, 4)
                rect = cv2.putText(img=rect, 
                                   text=str(f"{idx}"), 
                                   org=(rect_start[0], rect_start[1]+50), 
                                   fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                   fontScale=2, 
                                   thickness=2, 
                                   color=damage, 
                                   lineType=cv2.LINE_AA, 
                                   bottomLeftOrigin=False
                                   )
            
                imwrite(color_path, rect)

            else :
                color_recting = imread(color_path)
                color_recting = cv2.cvtColor(color_recting, cv2.COLOR_RGB2BGR)
                rect_start = (int(point[2]), int(point[0]))
                rect_end = (int(point[3]), int(point[1]))
                rect = cv2.rectangle(color_recting, rect_start, rect_end, damage, 4)
                rect = cv2.putText(img=rect, 
                                   text=str(f"{idx}"), 
                                   org=(rect_start[0], rect_start[1]+50), 
                                   fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                   fontScale=2, 
                                   thickness=2, 
                                   color=damage, 
                                   lineType=cv2.LINE_AA, 
                                   bottomLeftOrigin=False
                                   )
            
            
                imwrite(color_path, rect)


if __name__ == '__main__':
    main()