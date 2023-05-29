import os 
import argparse

from glob import glob
from tqdm import tqdm

import cv2
import numpy as np

from py_script.utils.utils import *


from skimage.measure import label, regionprops
import matplotlib.pyplot as plt 


parser = argparse.ArgumentParser()

parser.add_argument("source_path", help="file path to images to be cropped", type=str)
parser.add_argument("store_path", help="file path to store cropped images", type=str)
parser.add_argument("--dataset_type", default='cityscapes', type=str)
parser.add_argument("--target_class_num", default=1, type=int)
parser.add_argument("--save_src_img", default=None, type=bool)




args = parser.parse_args()

def main():

    source_path = args.source_path
    store_path = args.store_path
    dataset_type = args.dataset_type
    target_class_num = args.target_class_num

    color_save_path = os.path.join(store_path, "color")
    gtFine_save_path = os.path.join(store_path, "gtFine")
    leftImg8bit_save_path = os.path.join(store_path, "leftImg8bit")
    os.makedirs(color_save_path, exist_ok=True)
    os.makedirs(gtFine_save_path, exist_ok=True)
    os.makedirs(leftImg8bit_save_path, exist_ok=True)

    src_folder = glob(os.path.join(source_path, 'leftImg8bit', '*'))
    for f in src_folder:
        foldername = os.path.basename(f)
        os.makedirs(os.path.join(gtFine_save_path, foldername), exist_ok=True)
        os.makedirs(os.path.join(leftImg8bit_save_path, foldername), exist_ok=True)
        os.makedirs(os.path.join(color_save_path, foldername), exist_ok=True)

    if dataset_type == 'cityscapes' : 

        img_list = glob(os.path.join(source_path, 'leftImg8bit', '*', '*.png'))

        
        if not img_list :
            raise Exception("There is no image in the leftImg8bit folder.")

    # loop through all the imgs 
    for img_path in tqdm(img_list) :
        img = imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        palette = np.array([[0, 0, 0],
               [0, 255, 0],
               [0, 255, 0],
               [0, 255, 255], 
               [255, 0, 0]])

        gt_img_path = img_path.replace('\\leftImg8bit', '\\gtFine')
        gt_img_path = gt_img_path.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        gt_img = imread(gt_img_path)
        gt_img[gt_img != target_class_num] = 0
        gt_img[gt_img == target_class_num] = 1

        gt_img_target = gt_img == 1 
        gt_img_label = label(gt_img_target)

        if args.save_src_img == True :
            img_src_store_path = img_path.replace(source_path, store_path)
            imwrite(img_src_store_path, img)
            
            gt_path = img_src_store_path.replace('\\leftImg8bit', '\\gtFine')
            gt_path = gt_path.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            imwrite(gt_path, gt_img)
            
            color_path = img_src_store_path.replace('\\leftImg8bit', '\\color')
            color_path = color_path.replace('_leftImg8bit.png', '_color.png')
            colormap = blendImageWithColorMap(image=img, label=gt_img, palette=palette, alpha=float(0.5))
            imwrite(color_path, colormap)

        
        for region in regionprops(gt_img_label):
            # take regions with large enough areas
            if region.area >= 5000:   
                min_x, min_y, max_x, max_y = region.bbox

                crop_img = img[min_x:max_x, min_y:max_y, :]
                crop_gt = gt_img[min_x:max_x, min_y:max_y]
                
                
                
                img_store_path = img_path.replace(source_path, store_path)
                crop_img_store_path = img_store_path.replace(f'_leftImg8bit.png', f'_{target_class_num}_{min_x}_{min_y}_{max_x}_{max_y}_leftImg8bit.png')

                gt_store_path = crop_img_store_path.replace('\\leftImg8bit', '\\gtFine')
                crop_gt_store_path = gt_store_path.replace(f'_leftImg8bit.png', f'_gtFine_labelIds.png')

                
                
                imwrite(crop_img_store_path, crop_img)
                imwrite(crop_gt_store_path, crop_gt)

                color_store_path = crop_img_store_path.replace('\\leftImg8bit', '\\color')
                crop_color_store_path = color_store_path.replace("_leftImg8bit.png", "_color.png")
                
                crop_colormap = blendImageWithColorMap(image=crop_img, label=crop_gt, palette=palette, alpha=float(0.5))

                imwrite(crop_color_store_path, crop_colormap)

                

            
                

if __name__ == '__main__':
    

    main()
