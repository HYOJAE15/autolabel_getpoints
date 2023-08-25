import os 
import argparse

from glob import glob
from tqdm import tqdm

import cv2
import numpy as np

from py_script.utils.utils import blendImageWithColorMap

parser = argparse.ArgumentParser()

parser.add_argument("source_path", help="file path to images to be cropped", type=str)
parser.add_argument("store_path", help="file path to store cropped images", type=str)
parser.add_argument("--dataset_type", default='cityscapes', type=str)
parser.add_argument("--target_class_num", default=1, type=int)
parser.add_argument("--save_img_path", default=False, type=str)

"""
target class num
"""
# crack: 1
# efflorescence: 2
# rebar-exposure: 3
# spalling: 4

args = parser.parse_args()

def imread(path):
    
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    nparray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(nparray, cv2.IMREAD_UNCHANGED)

    return bgrImage

def imwrite(path, image):
    _, ext = os.path.splitext(path)
    cv2.imencode(ext, image)[1].tofile(path)

def main():

    source_path = args.source_path
    store_path = args.store_path
    dataset_type = args.dataset_type
    target_class_num = args.target_class_num
    save_img_path = args.save_img_path

    damage = [
        'BackGround',
        'Crack',
        'Efflorescence',
        'RebarExposure',
        'Spalling'
    ]

    palette = [
        np.array([[0, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 0, 255]]),
        np.array([[0, 0, 0], [0, 255, 0]]), 
        np.array([[0, 0, 0], [0, 255, 255]]), 
        np.array([[0, 0, 0], [255, 0, 0]])
        ]
    
    if dataset_type == 'cityscapes' : 

        img_list = glob(os.path.join(source_path, 'leftImg8bit', '*', '*.png'))

        if not img_list :
            raise Exception("There is no image in the leftImg8bit folder.")

    # loop through all the imgs 
    for img_path in tqdm(img_list) :
        img = imread(img_path)
        
        # window os.path.join separate is '\\'
        # ubuntu os.path.join seperate is '/'
        
        gt_img_path = img_path.replace('\\leftImg8bit', '\\gtFine')
        gt_img_path = gt_img_path.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        gt_img = imread(gt_img_path)
        gt_img[gt_img != target_class_num] = 0
        gt_img[gt_img == target_class_num] = 1

        
        
        if [1] in np.unique(gt_img) :
            
            img_store_path = img_path.replace(source_path, store_path)
            img_store_path = img_store_path.replace(f'_leftImg8bit.png', f'_{target_class_num}_leftImg8bit.png')

            gt_store_path = img_store_path.replace('\\leftImg8bit', '\\gtFine')
            gt_store_path = gt_store_path.replace(f'_leftImg8bit.png', f'_gtFine_labelIds.png')

            color_path = img_store_path.replace('\\leftImg8bit', '\\color')
            color_path = color_path.replace('_leftImg8bit.png', '_color.png')
            colormap = blendImageWithColorMap(image=img, 
                                              label=gt_img, 
                                              palette=palette[target_class_num], 
                                              alpha=float(0.5))
            
            if save_img_path != False:
                only_img_name_path = img_store_path.replace(store_path, save_img_path)
                only_img_name_path = only_img_name_path.replace('\\leftImg8bit', '')
                only_img_name_path = only_img_name_path.replace(f'_leftImg8bit.png', '.png')
                
                os.makedirs(os.path.dirname(only_img_name_path), exist_ok=True)
                
                imwrite(only_img_name_path, img)
             
            os.makedirs(os.path.dirname(img_store_path), exist_ok=True)
            os.makedirs(os.path.dirname(gt_store_path), exist_ok=True)
            os.makedirs(os.path.dirname(color_path), exist_ok=True)
            
            imwrite(img_store_path, img)
            imwrite(gt_store_path, gt_img)
            imwrite(color_path, colormap)
            
if __name__ == '__main__':
    

    main()
