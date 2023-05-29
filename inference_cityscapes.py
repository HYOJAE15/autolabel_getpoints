import argparse
import os 
from tqdm import tqdm
from glob import glob 
import sys

sys.path.append("./dnn/mmsegmentation")
from mmseg.apis import init_segmentor, inference_segmentor

from py_script.utils.utils import *

import cv2
import numpy as np


"""
콘크리트 손상 정보 inference 자료 cityscapes format 으로 생성
"""


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
        'save_dir',
        help=('directory to save result'))
    parser.add_argument(
        '--image_type',
        default="png",
        help=('image type, ["png", "tiff"]'))
    parser.add_argument(
        '--damage_type',
        default='crack',
        help=('damage type to be inference'))
    parser.add_argument(
        '--histogram_equalization_type',
        default=None,
        help=('histogram equalization type, ["gr", "hsv", "ycc"]'))
    parser.add_argument(
        '--erosion_dilation',
        default=None,
        help=('["er", "di"]'))
    
    

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

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device='cuda:0')

    img_list = glob(os.path.join(args.img_dir, f'*.{args.image_type}'))

    palette = np.array([[0, 0, 0],
               [0, 255, 0],
               [0, 255, 0],
               [0, 255, 255], 
               [255, 0, 0]])


    damage, damage_idx = get_damage_palette (args.damage_type)

    color_dir_path = os.path.join(args.save_dir, "color")
    os.makedirs(color_dir_path, exist_ok=True)
    

    for img in tqdm(img_list):
        
        gt_path = make_cityscapes_format_imagetype(
                                         image=img, 
                                         save_dir=args.save_dir, 
                                         image_type=args.image_type
                                        )

        src = imread(img)
        src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        
        histEqualization_type = args.histogram_equalization_type
        
        # histEqualization = histEqualization_gr(src) if histEqualization_type == 'gr' 
        
        if histEqualization_type == "gr" :
            src = histEqualization_gr(src)
            
        elif histEqualization_type == "hsv" :
            src = histEqualization_hsv(src)
            
        elif histEqualization_type == "ycc" :
            src = histEqualization_ycc(src)
            

        # cv2.imshow('src', src)
        # cv2.waitKey()
        
        src_label = imread(gt_path)
        
        result = inference_segmentor(model, src)

        idx = np.argwhere(result[0] == 1)
        y_idx, x_idx = idx[:, 0], idx[:, 1]
        src_label[y_idx, x_idx] = 1
        
        label_transform = args.erosion_dilation
        if label_transform == "er":
            kernel = np.ones((5, 5), np.uint8)
            src_label = cv2.erode(src_label, kernel, iterations=6)

        imwrite(gt_path, src_label)

        img_basename = os.path.basename(img)
        img_color_basename = img_basename.replace(f".{args.image_type}", "_color.png")
        color_path = os.path.join(color_dir_path, img_color_basename)

        colormap = blendImageWithColorMap(image=src, label=src_label, palette=palette, alpha=float(0.5))

        imwrite(color_path, colormap)
    
            
        
if __name__ == '__main__':
    main()
